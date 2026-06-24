"""
End-to-end training run:  python train.py
Wires the fraud/ functions together, prints metrics + top drivers, saves the model.
"""
import joblib

from fraud import (
    load_data, clean, stratified_split,
    train_model, evaluate, feature_importances,
)

TARGET = "is_fraud"


def main() -> None:
    df = clean(load_data("data/transactions.csv"))
    train_df, test_df = stratified_split(df, target=TARGET)

    # TODO: turn the cleaned frames into model-ready X/y (encode categoricals,
    # engineer features). Keep any fitting on the training split only.
    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    model = train_model(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    print("Metrics:", {k: round(v, 3) for k, v in metrics.items()})

    print("\nTop fraud drivers:")
    for feat, imp in feature_importances(model, list(X_train.columns))[:5]:
        print(f"  {feat:<24} {imp:+.4f}")

    joblib.dump(model, "fraud_model.joblib")
    print("\nSaved fraud_model.joblib")


if __name__ == "__main__":
    main()
