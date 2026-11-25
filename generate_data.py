import numpy as np
import pandas as pd

def generate_churn_dataset(n_samples=5000, seed=42):
    np.random.seed(seed)

    customer_id = [f"CUST_{10000 + i}" for i in range(n_samples)]
    tenure_months = np.random.randint(1, 48, size=n_samples)
    contract_type = np.random.choice(
        ["Month-to-Month", "Annual", "Two-Year"],
        size=n_samples,
        p=[0.55, 0.30, 0.15]
    )
    monthly_charges = np.round(np.random.uniform(1500, 12000, size=n_samples), 2)  # INR
    login_days_last_30 = np.random.randint(0, 31, size=n_samples)
    usage_drop_pct = np.round(np.random.uniform(-10, 80, size=n_samples), 1)
    support_tickets = np.random.poisson(lam=1.4, size=n_samples)
    late_payments = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.72, 0.18, 0.07, 0.03])

    # Calculate churn probability based on realistic business drivers
    logit = (
        -2.2
        + 0.035 * usage_drop_pct
        + 0.45 * support_tickets
        + 0.65 * late_payments
        - 0.04 * tenure_months
        - 0.05 * login_days_last_30
        + np.where(contract_type == "Month-to-Month", 0.95, 0.0)
        + np.where(contract_type == "Two-Year", -0.85, 0.0)
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_id,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges_inr": monthly_charges,
        "login_days_last_30": login_days_last_30,
        "usage_drop_pct": usage_drop_pct,
        "support_tickets": support_tickets,
        "late_payments": late_payments,
        "churn": churn
    })

    df.to_csv("customer_churn_data.csv", index=False)
    print(f"Dataset generated: {len(df)} rows | Churn Rate: {df['churn'].mean():.2%}")

if __name__ == "__main__":
    generate_churn_dataset()
