from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'data'
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)

TEXT_DATASET = DATA_DIR / 'text_dataset.csv'
TEXT_MODEL = MODELS_DIR / 'text_model.joblib'
BEHAVIOR_MODEL = MODELS_DIR / 'behavior_model.joblib'
METRICS_FILE = MODELS_DIR / 'metrics.json'
CONFUSION_FILE = MODELS_DIR / 'text_confusion_matrix.csv'
RANDOM_STATE = 42


def build_text_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ('word_tfidf', TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, max_features=20000, sublinear_tf=True)),
        ('char_tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, max_features=25000, sublinear_tf=True)),
    ])


def train_text_model() -> dict:
    df = pd.read_csv(TEXT_DATASET)
    df = df[['text', 'label']].copy()
    df['text'] = df['text'].astype(str).str.strip()
    df['label'] = df['label'].astype(str).str.strip()
    df = df[(df['text'] != '') & (df['label'] != '')].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
    )

    vectorizer = build_text_vectorizer()
    clf = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=RANDOM_STATE)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_test_vec)

    labels = sorted(df['label'].unique())
    confusion = confusion_matrix(y_test, preds, labels=labels)
    pd.DataFrame(confusion, index=[f'true_{x}' for x in labels], columns=[f'pred_{x}' for x in labels]).to_csv(CONFUSION_FILE)
    metrics = {
        'dataset_size': int(len(df)),
        'label_distribution': df['label'].value_counts().sort_index().to_dict(),
        'test_accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'test_macro_f1': round(float(f1_score(y_test, preds, average='macro')), 4),
        'classification_report': classification_report(y_test, preds, output_dict=True, zero_division=0),
    }

    joblib.dump({'vectorizer': vectorizer, 'model': clf, 'labels': labels}, TEXT_MODEL)
    return metrics


def generate_behavior_dataset(n_normal: int = 900, n_risky: int = 300) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)
    normal = pd.DataFrame({
        'gio_gui': np.random.randint(8, 18, n_normal),
        'la_ngoai_cong_ty': np.random.choice([0, 1], n_normal, p=[0.96, 0.04]),
        'dung_luong_mb': np.random.exponential(18, n_normal).clip(0.1, 90),
        'so_lan_gui': np.random.randint(1, 5, n_normal),
        'ip_la': np.random.choice([0, 1], n_normal, p=[0.97, 0.03]),
        'thiet_bi_la': np.random.choice([0, 1], n_normal, p=[0.97, 0.03]),
        'usb_copy': np.random.choice([0, 1], n_normal, p=[0.96, 0.04]),
        'label': 0,
    })
    risky = pd.DataFrame({
        'gio_gui': np.random.choice(list(range(0, 7)) + list(range(21, 24)), n_risky),
        'la_ngoai_cong_ty': np.random.choice([0, 1], n_risky, p=[0.1, 0.9]),
        'dung_luong_mb': np.random.uniform(200, 3000, n_risky),
        'so_lan_gui': np.random.randint(8, 30, n_risky),
        'ip_la': np.random.choice([0, 1], n_risky, p=[0.2, 0.8]),
        'thiet_bi_la': np.random.choice([0, 1], n_risky, p=[0.2, 0.8]),
        'usb_copy': np.random.choice([0, 1], n_risky, p=[0.35, 0.65]),
        'label': 1,
    })
    return pd.concat([normal, risky], ignore_index=True)


def train_behavior_model() -> dict:
    df = generate_behavior_dataset()
    feats = ['gio_gui', 'la_ngoai_cong_ty', 'dung_luong_mb', 'so_lan_gui', 'ip_la', 'thiet_bi_la', 'usb_copy']
    X_train, X_test, y_train, y_test = train_test_split(df[feats], df['label'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['label'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=180, class_weight='balanced', random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'dataset_size': int(len(df)),
        'test_accuracy': round(float(accuracy_score(y_test, preds)), 4),
        'test_macro_f1': round(float(f1_score(y_test, preds, average='macro')), 4),
        'feature_importances': {name: round(float(score), 4) for name, score in zip(feats, model.feature_importances_)},
        'avg_positive_probability': round(float(np.mean(probs)), 4),
    }

    joblib.dump({'scaler': scaler, 'model': model, 'features': feats}, BEHAVIOR_MODEL)
    return metrics


def main() -> None:
    text_metrics = train_text_model()
    behavior_metrics = train_behavior_model()
    with open(METRICS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'text_model': text_metrics, 'behavior_model': behavior_metrics}, f, ensure_ascii=False, indent=2)
    print('Saved:', TEXT_MODEL)
    print('Saved:', BEHAVIOR_MODEL)
    print('Saved:', METRICS_FILE)


if __name__ == '__main__':
    main()
