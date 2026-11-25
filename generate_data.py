"""Generate a deterministic synthetic subscription churn dataset."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_churn_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    if n_samples < 20:
        raise ValueError("n_samples must be at least 20 for a useful train/test split.")
    rng = np.random.default_rng(seed)
    customer_id = [f"CUST_{10000 + i}" for i in range(n_samples)]
    tenure_months = rng.integers(1, 49, size=n_samples)
    contract_type = rng.choice(["Month-to-Month", "Annual", "Two-Year"], size=n_samples,
                               p=[0.55, 0.30, 0.15])
    monthly_charges = np.round(rng.uniform(1500, 12000, size=n_samples), 2)
    login_days_last_30 = rng.integers(0, 31, size=n_samples)
    usage_drop_pct = np.round(rng.uniform(-10, 80, size=n_samples), 1)
    support_tickets = rng.poisson(lam=1.4, size=n_samples)
    late_payments = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.72, 0.18, 0.07, 0.03])
    logit = (-2.2 + 0.035 * usage_drop_pct + 0.45 * support_tickets + 0.65 * late_payments
             - 0.04 * tenure_months - 0.05 * login_days_last_30
             + np.where(contract_type == "Month-to-Month", 0.95, 0.0)
             + np.where(contract_type == "Two-Year", -0.85, 0.0))
    probability = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_samples) < probability).astype(int)
    return pd.DataFrame({"customer_id": customer_id, "tenure_months": tenure_months,
                         "contract_type": contract_type, "monthly_charges_inr": monthly_charges,
                         "login_days_last_30": login_days_last_30, "usage_drop_pct": usage_drop_pct,
                         "support_tickets": support_tickets, "late_payments": late_payments, "churn": churn})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("customer_churn_data.csv"))
    args = parser.parse_args()
    try:
        dataset = generate_churn_dataset(args.samples, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"Generated {len(dataset):,} rows at {args.output}")
    print(f"Synthetic churn rate: {dataset['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
