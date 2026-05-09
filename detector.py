from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


class DLPDetector:
    def __init__(self, models_dir: Path | None = None) -> None:
        models = models_dir or MODELS_DIR
        text_bundle = joblib.load(models / "text_model.joblib")
        behavior_bundle = joblib.load(models / "behavior_model.joblib")
        anomaly_bundle = joblib.load(models / "anomaly_models.joblib")

        self.vectorizer = text_bundle["vectorizer"]
        self.text_model = text_bundle["model"]
        self.text_model_name = text_bundle.get("best_model_name", "unknown")
        self.text_labels = text_bundle.get("labels", [])

        self.behavior_pipeline = behavior_bundle["pipeline"]
        self.behavior_features = behavior_bundle["features"]
        self.anomaly_models = anomaly_bundle["models"]

        metrics_path = models / "metrics.json"
        self.metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

        self.patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            "phone": re.compile(r"(?<!\d)(?:\+84|84|0)(?:[ .-]?\d){8,10}(?!\d)"),
            "cccd": re.compile(r"(?<!\d)(?:\d[ .-]?){12}(?!\d)"),
            "bank_account": re.compile(r"(?<!\d)(?:\d[ .-]?){8,16}(?!\d)"),
            "address": re.compile(
                r"(?i)(?:số nhà|địa chỉ|đ/c)\s*:?\s*[^\n,;]{4,80}"
                r"|\b\d+\s+(?:đường|đ\. |phố|ngõ|ngách|hẻm)\s+[^\n,;]{3,80}"
            ),
            "credential": re.compile(
                r"(?i)(?:password|mật khẩu|passwd|api[_ -]?key|token|secret[_ -]?key)\s*[:=]\s*\S+"
            ),
            "person": re.compile(
                r"(?:(?:anh|chị|ông|bà|em|khách hàng|họ tên|tên)\s+)"
                r"((?:[A-ZÀ-Ỹ][a-zà-ỹ]+)(?:\s+[A-ZÀ-Ỹ][a-zà-ỹ]+){0,3})"
            ),
        }
        self.mask_tokens = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "cccd": "[CCCD]",
            "bank_account": "[BANK_ACCOUNT]",
            "address": "[ADDRESS]",
            "credential": "[CREDENTIAL]",
            "person": "[PERSON]",
        }
        self.leak_keywords = [
            "gửi ra ngoài",
            "gmail cá nhân",
            "email cá nhân",
            "copy usb",
            "usb",
            "chia sẻ công khai",
            "upload",
            "google drive công khai",
            "telegram cá nhân",
            "máy tính ở nhà",
            "đối thủ",
            "copy dữ liệu",
            "personal email",
            "personal gmail",
            "private email",
            "send outside",
            "send externally",
            "external sharing",
            "share publicly",
            "public link",
            "public cloud",
            "upload to cloud",
            "upload to drive",
            "google drive",
            "dropbox",
            "onedrive",
            "telegram",
            "customer database",
            "customer list",
            "client list",
            "employee salary",
            "salary file",
            "confidential",
            "secret",
            "source code",
            "api key",
            "access token",
            "competitor",
            "work from home",
            "home computer",
        ]

    def _clean_num(self, value: str) -> str:
        return re.sub(r"\D", "", value)

    def _behavior_frame(self, behavior: dict[str, Any]) -> pd.DataFrame:
        defaults = {
            "gio_gui": 12,
            "la_ngoai_cong_ty": 0,
            "dung_luong_mb": 1.0,
            "so_lan_gui": 1,
            "ip_la": 0,
            "thiet_bi_la": 0,
            "usb_copy": 0,
            "channel": "Email",
            "sender_department": "IT",
            "receiver_domain": "internal",
            "file_type": "txt",
            "destination": "internal",
        }
        row = {**defaults, **behavior}
        return pd.DataFrame([row])[self.behavior_features]

    def detect_pii(self, text: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for key, pattern in self.patterns.items():
            vals = []
            for match in pattern.finditer(text):
                value = match.group(1) if key == "person" and match.groups() else match.group(0)
                if key in {"phone", "cccd", "bank_account"}:
                    value = self._clean_num(value)
                vals.append(value.strip())
            vals = sorted({v for v in vals if v})
            if vals:
                found[key] = vals

        if "bank_account" in found:
            phones = set(found.get("phone", []))
            cccds = set(found.get("cccd", []))
            found["bank_account"] = [
                x for x in found["bank_account"] if x not in phones and x not in cccds and len(x) >= 9
            ]
            if not found["bank_account"]:
                del found["bank_account"]
        return found

    def mask_pii(self, text: str) -> str:
        masked = text
        for key, pattern in self.patterns.items():
            if key == "person":
                masked = pattern.sub(lambda m: m.group(0).replace(m.group(1), self.mask_tokens[key]), masked)
            else:
                masked = pattern.sub(self.mask_tokens[key], masked)
        return masked

    def classify_text(self, text: str) -> tuple[str, dict[str, float], float]:
        X = self.vectorizer.transform([text])
        pred = self.text_model.predict(X)[0]
        if hasattr(self.text_model, "predict_proba"):
            probs_raw = self.text_model.predict_proba(X)[0]
            probs = {cls: float(score) for cls, score in zip(self.text_model.classes_, probs_raw)}
        else:
            probs = {label: 1.0 if label == pred else 0.0 for label in self.text_labels}
        return pred, probs, max(probs.values())

    def score_behavior(self, behavior: dict[str, Any]) -> tuple[float, dict[str, float]]:
        X = self._behavior_frame(behavior)
        prob = float(self.behavior_pipeline.predict_proba(X)[0][1])
        rf_model = self.behavior_pipeline.named_steps["model"]
        feature_names = self.behavior_pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = {
            name: float(score)
            for name, score in zip(feature_names, rf_model.feature_importances_)
        }
        return prob, importances

    def score_anomaly(self, behavior: dict[str, Any]) -> tuple[float, dict[str, float]]:
        X = self._behavior_frame(behavior)
        scores = {}
        risks = []
        for name, model in self.anomaly_models.items():
            score = float(model.decision_function(X)[0])
            risk = 1.0 / (1.0 + math.exp(4.0 * score))
            scores[name] = score
            scores[f"{name}_risk"] = risk
            risks.append(risk)
        return float(sum(risks) / len(risks)), scores

    def decide_action(
        self,
        doc_label: str,
        pii: dict[str, list[str]],
        behavior_prob: float,
        anomaly_risk: float,
        text: str,
        behavior: dict[str, Any],
    ) -> tuple[str, float, str]:
        doc_risk_map = {
            "safe": 0.10,
            "pii": 0.48,
            "credential": 0.72,
            "financial": 0.66,
            "confidential": 0.58,
            "sensitive": 0.56,
            "leak_risk": 0.86,
        }
        pii_count = sum(len(v) for v in pii.values())
        pii_score = min(
            1.0,
            (pii_count * 0.08)
            + (0.22 if "credential" in pii else 0)
            + (0.12 if "address" in pii else 0),
        )
        leak_context = 0.15 if any(k in text.lower() for k in self.leak_keywords) else 0.0
        channel_context = 0.0
        if behavior.get("destination") in {"personal_email", "public_cloud"}:
            channel_context += 0.08
        if behavior.get("receiver_domain") in {"gmail.com", "unknown", "public_cloud"}:
            channel_context += 0.06
        if behavior.get("file_type") in {"zip", "source", "image"}:
            channel_context += 0.04

        combined_behavior = behavior_prob * 0.65 + anomaly_risk * 0.35
        combined = min(
            1.0,
            doc_risk_map.get(doc_label, 0.2) * 0.38
            + combined_behavior * 0.32
            + pii_score * 0.18
            + leak_context
            + channel_context,
        )

        if combined >= 0.78:
            return "BLOCK", combined, "Chặn hoàn toàn vì dữ liệu nhạy cảm đi kèm hành vi có nguy cơ rò rỉ cao."
        if combined >= 0.58:
            return "QUARANTINE", combined, "Cách ly để quản trị viên kiểm tra thủ công trước khi cho phép gửi."
        if combined >= 0.42:
            return "ENCRYPT", combined, "Cho phép luồng dữ liệu tiếp tục nhưng yêu cầu mã hóa hoặc kênh gửi an toàn."
        if combined >= 0.28:
            return "WARN", combined, "Cảnh báo người dùng và ghi log để giám sát thêm."
        return "ALLOW", combined, "Cho phép vì chưa thấy tín hiệu rủi ro đáng kể."

    def analyze(self, text: str, behavior: dict[str, Any]) -> dict[str, Any]:
        pii = self.detect_pii(text)
        masked_text = self.mask_pii(text)
        doc_label, probs, confidence = self.classify_text(masked_text)
        behavior_prob, feature_importances = self.score_behavior(behavior)
        anomaly_risk, anomaly_scores = self.score_anomaly(behavior)
        action, score, reason = self.decide_action(
            doc_label,
            pii,
            behavior_prob,
            anomaly_risk,
            text,
            behavior,
        )
        return {
            "pii": pii,
            "masked_text": masked_text,
            "doc_label": doc_label,
            "doc_probabilities": probs,
            "doc_confidence": confidence,
            "text_model_name": self.text_model_name,
            "behavior_probability": behavior_prob,
            "anomaly_risk": anomaly_risk,
            "anomaly_scores": anomaly_scores,
            "feature_importances": feature_importances,
            "action": action,
            "dlp_score": score,
            "reason": reason,
        }
