import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def assign_retention_track(row):
    if row["late_payments"] > 0:
        return "Billing Friction", "Flexible billing date or 14-day grace window"
    if row["monthly_charges_inr"] >= 7500:
        return "Critical VIP", "1-on-1 onboarding review and workflow audit"
    return "Contract Exposure", "15% discount incentive to switch to Annual billing"


def run_pipeline():
    df = pd.read_csv("customer_churn_data.csv")

    X = pd.get_dummies(df.drop(columns=["customer_id", "churn"]), drop_first=True)
    y = df["churn"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= 0.5).astype(int)

    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_probs):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_preds))

    test_results = df.loc[idx_test].copy()
    test_results["churn_probability"] = y_probs.round(3)
    test_results["annual_revenue_at_risk_inr"] = (
        test_results["monthly_charges_inr"] * 12 * test_results["churn_probability"]
    ).round(2)

    high_risk = test_results[test_results["churn_probability"] >= 0.65].copy()
    tracks = high_risk.apply(assign_retention_track, axis=1)
    high_risk["risk_segment"] = [t[0] for t in tracks]
    high_risk["recommended_action"] = [t[1] for t in tracks]

    high_risk = high_risk.sort_values(
        by="annual_revenue_at_risk_inr", ascending=False
    )
    high_risk.to_csv("high_risk_customers.csv", index=False)

    total_risk = high_risk["annual_revenue_at_risk_inr"].sum()
    print(
        f"Flagged {len(high_risk)} high-risk accounts | Total Revenue at Risk: ₹{total_risk:,.2f}"
    )


if __name__ == "__main__":
    run_pipeline()
