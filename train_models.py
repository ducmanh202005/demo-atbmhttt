from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, OneClassSVM

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TEXT_DATASET = DATA_DIR / "text_dataset.csv"
FEEDBACK_FILE = DATA_DIR / "feedback.csv"
BEHAVIOR_DATASET = DATA_DIR / "behavior_dataset.csv"
TEXT_MODEL = MODELS_DIR / "text_model.joblib"
BEHAVIOR_MODEL = MODELS_DIR / "behavior_model.joblib"
ANOMALY_MODEL = MODELS_DIR / "anomaly_models.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
CONFUSION_FILE = MODELS_DIR / "text_confusion_matrix.csv"
RANDOM_STATE = 42
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "cp1252")

NUMERIC_BEHAVIOR_FEATURES = [
    "gio_gui",
    "la_ngoai_cong_ty",
    "dung_luong_mb",
    "so_lan_gui",
    "ip_la",
    "thiet_bi_la",
    "usb_copy",
]
CATEGORICAL_BEHAVIOR_FEATURES = [
    "channel",
    "sender_department",
    "receiver_domain",
    "file_type",
    "destination",
]
BEHAVIOR_FEATURES = NUMERIC_BEHAVIOR_FEATURES + CATEGORICAL_BEHAVIOR_FEATURES
TEXT_LABELS = {"safe", "pii", "credential", "financial", "confidential", "sensitive", "leak_risk"}


def read_csv_flexible(path: Path, **kwargs) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in CSV_ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=encoding, **kwargs)
            df.columns = [str(col).lstrip("\ufeff").strip() for col in df.columns]
            return df
        except UnicodeError as exc:
            last_error = exc
    raise UnicodeError(f"Không đọc được {path} bằng các encoding {CSV_ENCODINGS}") from last_error


def feedback_to_label(row: pd.Series) -> str | None:
    feedback = str(row.get("user_feedback", "")).strip()
    action = str(row.get("action", "")).strip()
    doc_label = str(row.get("doc_label", "")).strip()

    if feedback == "false_negative":
        return "leak_risk"
    if feedback == "valid_business_case":
        return "confidential"
    if feedback == "true_positive":
        if action in {"BLOCK", "QUARANTINE"}:
            return "leak_risk"
        if action in {"ENCRYPT", "WARN"}:
            return doc_label if doc_label in TEXT_LABELS - {"safe", "leak_risk"} else "confidential"
        if action == "ALLOW":
            return "safe"
    if feedback == "false_positive":
        if action in {"BLOCK", "QUARANTINE"}:
            return "confidential" if doc_label == "leak_risk" else "safe"
        if action in {"ENCRYPT", "WARN"}:
            return "safe"
    return None


def load_text_training_data() -> tuple[pd.DataFrame, int]:
    base = read_csv_flexible(TEXT_DATASET, sep=None, engine="python")
    base = base[["text", "label"]].copy()
    base["source"] = "base_dataset"

    if not FEEDBACK_FILE.exists() or FEEDBACK_FILE.stat().st_size == 0:
        return base, 0

    feedback = read_csv_flexible(FEEDBACK_FILE)
    required = {"text", "doc_label", "action", "user_feedback"}
    if not required.issubset(feedback.columns):
        return base, 0

    feedback = feedback.copy()
    feedback["label"] = feedback.apply(feedback_to_label, axis=1)
    feedback = feedback[["text", "label"]].dropna()
    feedback["text"] = feedback["text"].astype(str).str.strip()
    feedback["label"] = feedback["label"].astype(str).str.strip()
    feedback = feedback[(feedback["text"] != "") & (feedback["label"].isin(TEXT_LABELS))]
    feedback = feedback.drop_duplicates(subset=["text", "label"])
    feedback["source"] = "admin_feedback"

    if feedback.empty:
        return base, 0
    return pd.concat([base, feedback], ignore_index=True), int(len(feedback))


def calibrated_svm() -> CalibratedClassifierCV:
    svm = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE)
    try:
        return CalibratedClassifierCV(estimator=svm, cv=3)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=svm, cv=3)


def build_text_vectorizer() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=2,
                    max_features=20000,
                    sublinear_tf=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=25000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def train_text_model() -> dict:
    df, feedback_rows_used = load_text_training_data()
    df = df[["text", "label", "source"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    vectorizer = build_text_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "linear_svm": calibrated_svm(),
        "naive_bayes": MultinomialNB(),
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    labels = sorted(df["label"].unique())
    trained_models = {}
    all_metrics = {}
    best_name = ""
    best_score = -1.0

    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro")
        all_metrics[name] = {
            "test_accuracy": round(float(acc), 4),
            "test_macro_f1": round(float(macro_f1), 4),
        }
        trained_models[name] = model
        if macro_f1 > best_score:
            best_score = macro_f1
            best_name = name

    best_model = trained_models[best_name]
    best_preds = best_model.predict(X_test_vec)
    confusion = confusion_matrix(y_test, best_preds, labels=labels)
    pd.DataFrame(
        confusion,
        index=[f"true_{x}" for x in labels],
        columns=[f"pred_{x}" for x in labels],
    ).to_csv(CONFUSION_FILE)

    metrics = {
        "dataset_size": int(len(df)),
        "base_rows": int((df["source"] == "base_dataset").sum()),
        "feedback_rows_used": feedback_rows_used,
        "label_distribution": df["label"].value_counts().sort_index().to_dict(),
        "best_model_name": best_name,
        "candidate_models": all_metrics,
        "best_model_report": classification_report(
            y_test,
            best_preds,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": best_model,
            "best_model_name": best_name,
            "candidate_models": trained_models,
            "labels": labels,
        },
        TEXT_MODEL,
    )
    return metrics


def generate_behavior_dataset(n_normal: int = 1200, n_risky: int = 450) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    normal = pd.DataFrame(
        {
            "gio_gui": rng.integers(8, 18, n_normal),
            "la_ngoai_cong_ty": rng.choice([0, 1], n_normal, p=[0.94, 0.06]),
            "dung_luong_mb": rng.exponential(18, n_normal).clip(0.1, 95),
            "so_lan_gui": rng.integers(1, 5, n_normal),
            "ip_la": rng.choice([0, 1], n_normal, p=[0.97, 0.03]),
            "thiet_bi_la": rng.choice([0, 1], n_normal, p=[0.97, 0.03]),
            "usb_copy": rng.choice([0, 1], n_normal, p=[0.97, 0.03]),
            "channel": rng.choice(["Email", "SharePoint", "Endpoint", "Cloud"], n_normal, p=[0.45, 0.25, 0.2, 0.1]),
            "sender_department": rng.choice(["HR", "Finance", "IT", "Sales", "R&D"], n_normal),
            "receiver_domain": rng.choice(["internal", "partner.com"], n_normal, p=[0.88, 0.12]),
            "file_type": rng.choice(["txt", "pdf", "docx", "xlsx"], n_normal, p=[0.12, 0.32, 0.34, 0.22]),
            "destination": rng.choice(["internal", "partner"], n_normal, p=[0.9, 0.1]),
            "label": 0,
        }
    )

    risky = pd.DataFrame(
        {
            "gio_gui": rng.choice(list(range(0, 7)) + list(range(21, 24)), n_risky),
            "la_ngoai_cong_ty": rng.choice([0, 1], n_risky, p=[0.08, 0.92]),
            "dung_luong_mb": rng.uniform(180, 3200, n_risky),
            "so_lan_gui": rng.integers(8, 35, n_risky),
            "ip_la": rng.choice([0, 1], n_risky, p=[0.22, 0.78]),
            "thiet_bi_la": rng.choice([0, 1], n_risky, p=[0.25, 0.75]),
            "usb_copy": rng.choice([0, 1], n_risky, p=[0.32, 0.68]),
            "channel": rng.choice(["Email", "USB", "Cloud", "Endpoint"], n_risky, p=[0.35, 0.25, 0.3, 0.1]),
            "sender_department": rng.choice(["HR", "Finance", "IT", "Sales", "R&D"], n_risky),
            "receiver_domain": rng.choice(["gmail.com", "unknown", "public_cloud", "partner.com"], n_risky, p=[0.42, 0.25, 0.22, 0.11]),
            "file_type": rng.choice(["zip", "xlsx", "pdf", "image", "source"], n_risky, p=[0.27, 0.28, 0.18, 0.12, 0.15]),
            "destination": rng.choice(["personal_email", "public_cloud", "external", "partner"], n_risky, p=[0.4, 0.28, 0.22, 0.1]),
            "label": 1,
        }
    )
    return pd.concat([normal, risky], ignore_index=True)


def load_behavior_training_data() -> tuple[pd.DataFrame, str]:
    if not BEHAVIOR_DATASET.exists() or BEHAVIOR_DATASET.stat().st_size == 0:
        return generate_behavior_dataset(), "synthetic"

    df = read_csv_flexible(BEHAVIOR_DATASET, sep=None, engine="python")
    required = set(BEHAVIOR_FEATURES + ["label"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"behavior_dataset.csv thiếu cột: {sorted(missing)}")

    df = df[BEHAVIOR_FEATURES + ["label"]].copy()
    for col in NUMERIC_BEHAVIOR_FEATURES + ["label"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_BEHAVIOR_FEATURES:
        df[col] = df[col].astype(str).str.strip()

    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    label_counts = df["label"].value_counts().to_dict()
    if len(df) < 20 or label_counts.get(0, 0) < 5 or label_counts.get(1, 0) < 5:
        return generate_behavior_dataset(), "synthetic_fallback_not_enough_real_rows"

    return df, "behavior_dataset.csv"


def behavior_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_BEHAVIOR_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_BEHAVIOR_FEATURES),
        ]
    )


def train_behavior_model() -> dict:
    df, dataset_source = load_behavior_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[BEHAVIOR_FEATURES],
        df["label"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    rf_pipeline = Pipeline(
        [
            ("preprocess", behavior_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=220,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    preds = rf_pipeline.predict(X_test)
    probs = rf_pipeline.predict_proba(X_test)[:, 1]

    normal_train = X_train[y_train == 0]
    anomaly_models = {
        "isolation_forest": Pipeline(
            [
                ("preprocess", behavior_preprocessor()),
                ("model", IsolationForest(contamination=0.12, random_state=RANDOM_STATE)),
            ]
        ),
        "one_class_svm": Pipeline(
            [
                ("preprocess", behavior_preprocessor()),
                ("model", OneClassSVM(kernel="rbf", gamma="scale", nu=0.08)),
            ]
        ),
        "local_outlier_factor": Pipeline(
            [
                ("preprocess", behavior_preprocessor()),
                ("model", LocalOutlierFactor(n_neighbors=35, contamination=0.12, novelty=True)),
            ]
        ),
    }
    for model in anomaly_models.values():
        model.fit(normal_train)

    rf_model = rf_pipeline.named_steps["model"]
    feature_names = rf_pipeline.named_steps["preprocess"].get_feature_names_out()

    metrics = {
        "dataset_source": dataset_source,
        "dataset_size": int(len(df)),
        "label_distribution": df["label"].value_counts().sort_index().to_dict(),
        "test_accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "test_macro_f1": round(float(f1_score(y_test, preds, average="macro")), 4),
        "avg_positive_probability": round(float(np.mean(probs)), 4),
        "feature_importances": {
            name: round(float(score), 4)
            for name, score in zip(feature_names, rf_model.feature_importances_)
        },
        "anomaly_models": list(anomaly_models.keys()),
    }

    joblib.dump(
        {
            "pipeline": rf_pipeline,
            "features": BEHAVIOR_FEATURES,
            "numeric_features": NUMERIC_BEHAVIOR_FEATURES,
            "categorical_features": CATEGORICAL_BEHAVIOR_FEATURES,
        },
        BEHAVIOR_MODEL,
    )
    joblib.dump({"models": anomaly_models, "features": BEHAVIOR_FEATURES}, ANOMALY_MODEL)
    return metrics


def main() -> None:
    text_metrics = train_text_model()
    behavior_metrics = train_behavior_model()
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"text_model": text_metrics, "behavior_model": behavior_metrics},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("Saved:", TEXT_MODEL)
    print("Saved:", BEHAVIOR_MODEL)
    print("Saved:", ANOMALY_MODEL)
    print("Saved:", METRICS_FILE)


if __name__ == "__main__":
    main()
