"""Contract tests — fail against the starter stubs; make them pass.

Tiny inline data, fully offline. Only scikit-learn / numpy / pandas are used
here on purpose — do NOT import xgboost or hit the network in the tests.
"""
import numpy as np
import pandas as pd

from fraud import clean, stratified_split, evaluate


def _toy(n=600):
    """A tiny, clearly-separable fraud-like frame with dirt and rare positives."""
    rng = np.random.default_rng(0)
    amount = rng.uniform(1, 500, n).round(2)
    is_new_device = rng.integers(0, 2, n)
    # ~10% fraud here (still imbalanced, but enough positives for a stable split)
    y = ((is_new_device == 1) & (amount > 250)).astype(int)
    df = pd.DataFrame({
        "transaction_id": [f"T{i}" for i in range(n)],
        "amount": amount,
        "hour_of_day": rng.integers(0, 24, n),
        # mixed-case country codes — dirt to normalise
        "country_code": rng.choice(["au", "AU", "Us", "us", "GB"], n),
        "is_foreign": rng.integers(0, 2, n),
        "account_age_days": rng.integers(1, 3650, n),
        "txns_last_1h": rng.integers(0, 6, n),
        "is_new_device": is_new_device,
        "merchant_category": rng.choice(["dining", "fuel", None], n),
        "is_fraud": y,
    })
    return df


def test_clean_fills_category_and_normalises_country():
    out = clean(_toy())
    # missing merchant_category must be filled (no NaNs / no None left)
    assert out["merchant_category"].isna().sum() == 0
    # country_code must collapse to a single (upper) case: 'au' and 'AU' -> 'AU'
    codes = set(out["country_code"].unique())
    assert codes == {c.upper() for c in codes}
    assert "au" not in codes and "us" not in codes


def test_stratified_split_preserves_fraud_ratio():
    df = _toy()
    train_df, test_df = stratified_split(df, target="is_fraud", test_size=0.25,
                                         random_state=42)
    assert len(train_df) + len(test_df) == len(df)
    full = df["is_fraud"].mean()
    # stratification keeps the rare-class rate close on both sides
    assert abs(train_df["is_fraud"].mean() - full) < 0.02
    assert abs(test_df["is_fraud"].mean() - full) < 0.02


def test_evaluate_returns_metric_keys():
    # A trivial linear-separable problem so any sane model scores well.
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x": np.r_[rng.normal(0, 1, 200), rng.normal(6, 1, 200)]})
    y = pd.Series([0] * 200 + [1] * 200)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression().fit(X, y)

    m = evaluate(model, X, y)
    assert {"roc_auc", "precision", "recall"} <= set(m)
    assert m["roc_auc"] >= 0.6
