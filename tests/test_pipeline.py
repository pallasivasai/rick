"""Automated tests for the churn prediction workflow."""
import sys
from pathlib import Path

import pandas as pd
import pytest

# Make repository-root modules importable when pytest runs from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate_data import generate_churn_dataset
from churn_model import load_dataset, score_customers, train_model


def test_generator_is_reproducible():
    pd.testing.assert_frame_equal(generate_churn_dataset(100, 7), generate_churn_dataset(100, 7))


def test_generator_rejects_too_small_dataset():
    with pytest.raises(ValueError):
        generate_churn_dataset(10)


def test_load_dataset_reports_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="generate_data.py"):
        load_dataset(tmp_path / "missing.csv")


def test_training_and_scoring_produce_valid_results():
    data = generate_churn_dataset(400, 11)
    trained = train_model(data, threshold=0.6)
    scores = score_customers(data, trained["model"], threshold=0.6)
    assert 0 <= trained["metrics"]["roc_auc"] <= 1
    assert scores["churn_probability"].between(0, 1).all()
    assert {"risk_level", "annual_revenue_at_risk_inr", "recommended_action"} <= set(scores)
    assert scores["annual_revenue_at_risk_inr"].is_monotonic_decreasing


def test_load_dataset_validates_schema(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"customer_id": ["A"], "churn": [1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
