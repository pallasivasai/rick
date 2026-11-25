# Customer Churn Prediction & Revenue-Risk Analytics

A reproducible machine-learning project that estimates customer churn probability, translates risk into annual revenue exposure, and generates prioritized retention recommendations.

> **Data notice:** The generator creates synthetic demonstration data. Results on synthetic data are not evidence of real-world business performance. Validate using representative, time-aware production data before making customer decisions.

## Capabilities

- Deterministic synthetic dataset generation.
- Leakage-safe preprocessing and Random Forest training in one scikit-learn pipeline.
- Stratified holdout evaluation with ROC-AUC, precision, recall, accuracy, and confusion matrix.
- Customer scoring with Low / Medium / High risk levels.
- Annual revenue exposure estimates and prioritized retention suggestions.
- Exported model, metrics, full score list, and high-risk customer list.
- Automated tests and GitHub Actions CI across Python 3.10–3.12.

## Project structure

```text
rick/
├── .github/workflows/ci.yml
├── tests/test_pipeline.py
├── .gitignore
├── churn_model.py
├── generate_data.py
├── requirements.txt
└── README.md
```

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python generate_data.py
python churn_model.py
```

To customize the dataset, output directory, or decision threshold:

```bash
python generate_data.py --samples 8000 --seed 21 --output data/customers.csv
python churn_model.py --data data/customers.csv --output-dir outputs/run-01 --threshold 0.60
```

## Generated artifacts

| Artifact | Description |
| --- | --- |
| `churn_model.joblib` | Trained preprocessing and classifier pipeline with feature and threshold metadata |
| `metrics.json` | Holdout ROC-AUC, accuracy, precision, recall, confusion matrix, and class report |
| `customer_churn_scores.csv` | All customers ranked by estimated annual revenue exposure |
| `high_risk_customers.csv` | Customers at or above the configured probability threshold |

## Model methodology

Categorical values are one-hot encoded with unknown-category support. Numeric fields are median-imputed and standardized. Preprocessing is fitted only on the training split by keeping transformations and the classifier in a single pipeline.

The project uses a stratified 80/20 train/test split and a fixed random seed for reproducibility. The default risk threshold is 0.65. Tune the threshold using validation data, intervention capacity, and the relative cost of false positives and false negatives; it is not a universal business setting.

### Revenue exposure estimate

`annual_revenue_at_risk_inr = monthly_charges_inr × 12 × churn_probability`

This represents expected exposure, not guaranteed revenue loss or retained revenue. It does not model gross margin, contract duration, probability of successful intervention, or recovery outcomes.

### Retention recommendations

Illustrative recommendations are based on payment history, monthly charges, and contract type. They are decision-support suggestions only; require human review before customer outreach or incentives.

## Data dictionary

| Feature | Meaning |
| --- | --- |
| `customer_id` | Synthetic customer identifier |
| `tenure_months` | Customer tenure in months |
| `contract_type` | Month-to-Month, Annual, or Two-Year |
| `monthly_charges_inr` | Monthly subscription amount in INR |
| `login_days_last_30` | Days active in the last 30 days |
| `usage_drop_pct` | Simulated usage decline percentage |
| `support_tickets` | Support tickets recorded |
| `late_payments` | Number of late payments |
| `churn` | Target label: 1 = churn, 0 = retained |

## Tests

```bash
python -m pip install pytest
pytest -q
```

GitHub Actions runs tests on pushes and pull requests against Python 3.10, 3.11, and 3.12.

## Responsible use and production checklist

- Never present synthetic-data performance as a real customer result.
- Evaluate with representative, time-aware data and monitor data drift.
- Check probability calibration, subgroup performance, and error costs.
- Protect customer information; do not commit personally identifiable data.
- Keep retention decisions explainable and subject to human review.

## License

No license is currently specified. Add a license before granting reuse or redistribution permissions.
