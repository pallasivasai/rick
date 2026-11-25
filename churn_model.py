"""Train, evaluate, and score customer churn with a reproducible pipeline."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("churn")
RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.65
TARGET = "churn"
ID_COLUMN = "customer_id"
REQUIRED_COLUMNS = {ID_COLUMN, "tenure_months", "contract_type", "monthly_charges_inr",
                    "login_days_last_30", "usage_drop_pct", "support_tickets", "late_payments", TARGET}


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run 'python generate_data.py' first.")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if df.empty:
        raise ValueError("Dataset is empty.")
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target 'churn' must contain non-null binary labels 0 and 1.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Training requires both churn classes (0 and 1).")
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                                 ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                      ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    preprocessing = ColumnTransformer([("numeric", numeric_pipeline, numeric),
                                       ("categorical", categorical_pipeline, categorical)])
    classifier = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    return Pipeline([("preprocessing", preprocessing), ("classifier", classifier)])


def assign_retention_track(row: pd.Series) -> tuple[str, str]:
    if row["late_payments"] > 0:
        return "Billing Friction", "Review payment failures and offer a suitable billing arrangement"
    if row["monthly_charges_inr"] >= 7500:
        return "High-Value Account", "Schedule a customer success review and workflow audit"
    if row["contract_type"] == "Month-to-Month":
        return "Contract Exposure", "Discuss plan fit and annual-plan options without pressure"
    return "Engagement Support", "Review product adoption and offer targeted onboarding"


def train_model(df: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    if not 0 < threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")
    feature_columns = [c for c in df.columns if c not in {ID_COLUMN, TARGET}]
    X, y = df[feature_columns], df[TARGET].astype(int)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    model = build_pipeline(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "threshold": threshold, "train_rows": int(len(idx_train)), "test_rows": int(len(idx_test)),
        "churn_rate": float(y.mean()), "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0)}
    return {"model": model, "metrics": metrics, "test_indices": idx_test, "features": feature_columns}


def score_customers(df: pd.DataFrame, model: Pipeline, threshold: float) -> pd.DataFrame:
    expected = list(model.feature_names_in_)
    missing = sorted(set(expected).difference(df.columns))
    if missing:
        raise ValueError(f"Scoring data is missing features: {', '.join(missing)}")
    result = df.copy()
    probabilities = model.predict_proba(result[expected])[:, 1]
    result["churn_probability"] = probabilities.round(4)
    result["risk_level"] = pd.cut(probabilities, bins=[-0.001, 0.35, threshold, 1.0],
                                  labels=["Low", "Medium", "High"], include_lowest=True).astype(str)
    result["annual_revenue_at_risk_inr"] = (result["monthly_charges_inr"] * 12 * probabilities).round(2)
    tracks = result.apply(assign_retention_track, axis=1)
    result["retention_track"] = [item[0] for item in tracks]
    result["recommended_action"] = [item[1] for item in tracks]
    return result.sort_values(["annual_revenue_at_risk_inr", "churn_probability"], ascending=False)


def run_pipeline(data_path: Path, output_dir: Path, threshold: float) -> dict[str, Any]:
    df = load_dataset(data_path)
    trained = train_model(df, threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path, metrics_path = output_dir / "churn_model.joblib", output_dir / "metrics.json"
    scored_path, high_risk_path = output_dir / "customer_churn_scores.csv", output_dir / "high_risk_customers.csv"
    joblib.dump({"model": trained["model"], "threshold": threshold,
                 "features": trained["features"], "version": 1}, model_path)
    metrics_path.write_text(json.dumps(trained["metrics"], indent=2), encoding="utf-8")
    scored = score_customers(df, trained["model"], threshold)
    scored.to_csv(scored_path, index=False)
    scored.loc[scored["churn_probability"] >= threshold].to_csv(high_risk_path, index=False)
    risk = scored.loc[scored["churn_probability"] >= threshold, "annual_revenue_at_risk_inr"].sum()
    LOGGER.info("ROC-AUC %.4f | precision %.4f | recall %.4f",
                trained["metrics"]["roc_auc"], trained["metrics"]["precision"], trained["metrics"]["recall"])
    LOGGER.info("Scored %d customers; %d above threshold %.2f",
                len(scored), int((scored["churn_probability"] >= threshold).sum()), threshold)
    LOGGER.info("Estimated annual revenue exposure: ₹%s", f"{risk:,.2f}")
    LOGGER.info("Artifacts saved to %s", output_dir.resolve())
    return {"metrics": trained["metrics"], "model": model_path, "scores": scored_path,
            "high_risk": high_risk_path, "metrics_file": metrics_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("customer_churn_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    try:
        run_pipeline(args.data, args.output_dir, args.threshold)
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
