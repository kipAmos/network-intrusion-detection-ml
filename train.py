from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
TARGET_CANDIDATES = ("label", "class", "attack", "target")
# These fields identify a particular endpoint/session rather than traffic behaviour.
IDENTIFIER_COLUMNS = {
    "flow id", "src ip", "source ip", "dst ip", "destination ip",
    "timestamp", "source port", "destination port"
}


def clean_column_name(name: object) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\ufeff", "")).strip()


def load_dataset(path: Path, target_arg: str | None) -> tuple[pd.DataFrame, pd.Series, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\nPlace your labelled CSV in data/ or pass --data PATH."
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = [clean_column_name(c) for c in df.columns]
    df = df.loc[:, ~df.columns.duplicated()]
    target_lookup = {c.lower(): c for c in df.columns}

    if target_arg:
        target = target_lookup.get(target_arg.lower())
        if target is None:
            raise ValueError(f"Target column '{target_arg}' not found. Columns: {list(df.columns)}")
    else:
        target = next((target_lookup[c] for c in TARGET_CANDIDATES if c in target_lookup), None)
        if target is None:
            raise ValueError(
                "Could not identify the label column. Pass its name using --target, "
                "for example: --target Label"
            )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target])
    df[target] = df[target].astype(str).str.strip()
    df = df[df[target] != ""]
    df = df.drop_duplicates()

    y = df[target]
    X = df.drop(columns=[target])

    # Remove columns that can cause memorisation and poor generalisation.
    drop_cols = [c for c in X.columns if c.lower() in IDENTIFIER_COLUMNS]
    X = X.drop(columns=drop_cols, errors="ignore")

    # Convert numeric-looking text columns to numbers where possible.
    for col in X.columns:
        if X[col].dtype == "object":
            converted = pd.to_numeric(X[col].astype(str).str.strip(), errors="coerce")
            if converted.notna().mean() >= 0.90:
                X[col] = converted

    # Drop columns that contain no usable information.
    X = X.dropna(axis=1, how="all")
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols, errors="ignore")

    if X.empty:
        raise ValueError("No usable feature columns remain after cleaning.")
    if y.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")

    return X, y, target


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler())
        ])
        transformers.append(("numeric", numeric_pipe, numeric_cols))

    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def make_models(seed: int, n_classes: int) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install dependencies with: pip install -r requirements.txt"
        ) from exc

    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=seed, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, random_state=seed, n_jobs=-1,
            class_weight="balanced_subsample"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9,
            objective=("binary:logistic" if n_classes == 2 else "multi:softprob"),
            num_class=None if n_classes == 2 else n_classes,
            eval_metric="logloss" if n_classes == 2 else "mlogloss",
            tree_method="hist", random_state=seed, n_jobs=-1
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate network intrusion classifiers.")
    parser.add_argument("--data", default="data/network_traffic.csv", help="Path to labelled CSV dataset.")
    parser.add_argument("--target", default=None, help="Target/label column; auto-detected if omitted.")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=10, help="Stratified CV folds (default: 10).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        raise ValueError("--test-size must be between 0.1 and 0.4.")
    if args.cv < 2:
        raise ValueError("--cv must be at least 2.")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path

    print(f"Loading dataset: {data_path}")
    X, y_text, target_name = load_dataset(data_path, args.target)
    print(f"Rows: {len(X):,} | Features: {X.shape[1]} | Classes: {y_text.nunique()}")
    print("Class distribution:")
    print(y_text.value_counts().to_string())

    # Encode labels once so every model receives the same class IDs.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    class_counts = pd.Series(y).value_counts()
    folds = min(args.cv, int(class_counts.min()))
    if folds < 2:
        raise ValueError("Each class needs at least two records for stratified validation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    models = make_models(args.seed, len(label_encoder.classes_))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=args.seed)
    results = []
    fitted = {}

    for name, estimator in models.items():
        print(f"\nEvaluating {name} with {folds}-fold stratified cross-validation...")
        pipeline = Pipeline([
            ("preprocessing", build_preprocessor(X_train)),
            ("classifier", estimator)
        ])
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=1
        )
        print(f"Weighted F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        results.append({
            "model": name,
            "cv_weighted_f1_mean": float(scores.mean()),
            "cv_weighted_f1_std": float(scores.std())
        })
        fitted[name] = pipeline

    # Select the model with the strongest mean cross-validation weighted F1.
    results.sort(key=lambda item: item["cv_weighted_f1_mean"], reverse=True)
    best_name = results[0]["model"]
    best_pipeline = fitted[best_name]
    print(f"\nBest model by CV weighted F1: {best_name}")
    best_pipeline.fit(X_train, y_train)

    predictions = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_test, predictions)
    # Micro-averaged one-vs-rest false-positive rate.
    tp = np.trace(cm)
    fp = cm.sum(axis=0).sum() - tp
    fn = cm.sum(axis=1).sum() - tp
    # Aggregate the one-vs-rest tables for all classes.
    total_binary_decisions = len(y_test) * len(class_names)
    tn = total_binary_decisions - tp - fp - fn
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0

    try:
        probabilities = best_pipeline.predict_proba(X_test)
        auc = float(roc_auc_score(
            y_test, probabilities, multi_class="ovr", average="weighted"
        ))
    except (ValueError, AttributeError):
        auc = None

    class_names = [str(c) for c in label_encoder.classes_]
    report = classification_report(
        y_test, predictions, labels=np.arange(len(class_names)),
        target_names=class_names, output_dict=True, zero_division=0
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / "best_intrusion_model.joblib"
    joblib.dump({
        "pipeline": best_pipeline,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "target_column": target_name,
        "model_name": best_name
    }, model_path)

    metrics = {
        "dataset": str(data_path.name),
        "records_after_cleaning": int(len(X)),
        "feature_count": int(X.shape[1]),
        "class_names": class_names,
        "best_model": best_name,
        "cross_validation": results,
        "test_metrics": {
            "accuracy": float(accuracy),
            "weighted_precision": float(precision),
            "weighted_recall": float(recall),
            "weighted_f1": float(f1),
            "micro_false_positive_rate": fpr,
            "weighted_ovr_roc_auc": auc
        },
        "classification_report": report,
        "confusion_matrix": cm.tolist()
    }

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        REPORT_DIR / "confusion_matrix.csv"
    )
    pd.DataFrame(results).to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    print("\n===== HELD-OUT TEST RESULTS =====")
    print(f"Selected model: {best_name}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print(f"F1-score:  {f1:.4f} (weighted)")
    print(f"FPR:       {fpr:.4f} (micro)")
    print(f"ROC-AUC:   {auc if auc is not None else 'Not available'}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, predictions, labels=np.arange(len(class_names)),
        target_names=class_names, zero_division=0
    ))
    print(f"\nSaved model: {model_path}")
    print(f"Saved reports in: {REPORT_DIR}")


if __name__ == "__main__":
    main()
