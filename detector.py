from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / 'models'


class DLPDetector:
    def __init__(self, models_dir: Path | None = None) -> None:
        models = models_dir or MODELS_DIR
        text_bundle = joblib.load(models / 'text_model.joblib')
        behavior_bundle = joblib.load(models / 'behavior_model.joblib')
        self.vectorizer = text_bundle['vectorizer']
        self.text_model = text_bundle['model']
        self.text_labels = text_bundle.get('labels', [])
        self.behavior_scaler = behavior_bundle['scaler']
        self.behavior_model = behavior_bundle['model']
        self.behavior_features = behavior_bundle['features']
        metrics_path = models / 'metrics.json'
        self.metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path.exists() else {}

        self.patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
            'phone': re.compile(r'(?<!\d)(?:\+84|84|0)(?:[ .-]?\d){8,10}(?!\d)'),
            'cccd': re.compile(r'(?<!\d)(?:\d[ .-]?){12}(?!\d)'),
            'bank_account': re.compile(r'(?<!\d)(?:\d[ .-]?){8,16}(?!\d)'),
            'address': re.compile(r'(?i)(?:số nhà|địa chỉ|đ/c)\s*:?\s*[^\n,;]{4,80}|\b\d+\s+(?:đường|đ\. |phố|ngõ|ngách|hẻm)\s+[^\n,;]{3,80}'),
            'credential': re.compile(r'(?i)(?:password|mật khẩu|passwd|api[_ -]?key|token|secret[_ -]?key)\s*[:=]\s*\S+'),
            'person': re.compile(r'(?:(?:anh|chị|ông|bà|em|khách hàng|họ tên|tên)\s+)((?:[A-ZÀ-Ỹ][a-zà-ỹ]+)(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,3})'),
        }
        self.leak_keywords = [
            'gửi ra ngoài', 'gmail cá nhân', 'email cá nhân', 'copy usb', 'usb', 'chia sẻ công khai',
            'upload', 'google drive công khai', 'telegram cá nhân', 'máy tính ở nhà', 'đối thủ', 'copy dữ liệu',
        ]

    def _clean_num(self, value: str) -> str:
        return re.sub(r'\D', '', value)

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key, pattern in self.patterns.items():
            vals = []
            for m in pattern.finditer(text):
                v = m.group(1) if key == 'person' and m.groups() else m.group(0)
                if key in {'phone', 'cccd', 'bank_account'}:
                    v = self._clean_num(v)
                vals.append(v.strip())
            vals = sorted({v for v in vals if v})
            if vals:
                found[key] = vals
        # avoid bank_account swallowing phone/cccd
        if 'bank_account' in found:
            phones = set(found.get('phone', []))
            cccds = set(found.get('cccd', []))
            found['bank_account'] = [x for x in found['bank_account'] if x not in phones and x not in cccds and len(x) >= 9]
            if not found['bank_account']:
                del found['bank_account']
        return found

    def classify_text(self, text: str) -> tuple[str, dict[str, float], float]:
        X = self.vectorizer.transform([text])
        pred = self.text_model.predict(X)[0]
        probs_raw = self.text_model.predict_proba(X)[0]
        probs = {cls: float(score) for cls, score in zip(self.text_model.classes_, probs_raw)}
        return pred, probs, max(probs.values())

    def score_behavior(self, behavior: dict[str, Any]) -> tuple[float, dict[str, float]]:
        X = pd.DataFrame([behavior])[self.behavior_features]
        Xs = self.behavior_scaler.transform(X)
        prob = float(self.behavior_model.predict_proba(Xs)[0][1])
        importances = {name: float(score) for name, score in zip(self.behavior_features, self.behavior_model.feature_importances_)}
        return prob, importances

    def decide_action(self, doc_label: str, pii: dict[str, list[str]], behavior_prob: float, text: str) -> tuple[str, float, str]:
        doc_risk_map = {'safe': 0.15, 'sensitive': 0.58, 'leak_risk': 0.85}
        pii_score = min(1.0, (sum(len(v) for v in pii.values()) * 0.08) + (0.22 if 'credential' in pii else 0) + (0.12 if 'address' in pii else 0))
        leak_context = 0.15 if any(k in text.lower() for k in self.leak_keywords) else 0.0
        combined = min(1.0, doc_risk_map.get(doc_label, 0.2) * 0.45 + behavior_prob * 0.35 + pii_score * 0.20 + leak_context)
        if combined >= 0.78:
            return 'BLOCK', combined, 'Chặn hoàn toàn vì tài liệu nhạy cảm đi kèm hành vi có nguy cơ rò rỉ cao.'
        if combined >= 0.56:
            return 'QUARANTINE', combined, 'Cách ly để kiểm tra thủ công trước khi cho phép gửi.'
        if combined >= 0.32:
            return 'WARN', combined, 'Cảnh báo người dùng và ghi log để giám sát thêm.'
        return 'ALLOW', combined, 'Cho phép vì chưa thấy tín hiệu rủi ro đáng kể.'

    def analyze(self, text: str, behavior: dict[str, Any]) -> dict[str, Any]:
        pii = self.detect_pii(text)
        doc_label, probs, confidence = self.classify_text(text)
        behavior_prob, feature_importances = self.score_behavior(behavior)
        action, score, reason = self.decide_action(doc_label, pii, behavior_prob, text)
        return {
            'pii': pii,
            'doc_label': doc_label,
            'doc_probabilities': probs,
            'doc_confidence': confidence,
            'behavior_probability': behavior_prob,
            'feature_importances': feature_importances,
            'action': action,
            'dlp_score': score,
            'reason': reason,
        }
