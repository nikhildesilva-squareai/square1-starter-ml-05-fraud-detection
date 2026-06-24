"""
Modelling + evaluation. Baseline with logistic regression, then train XGBoost
(handle the imbalance with scale_pos_weight / class weights). Tests define the
contract for `evaluate`.
"""
from __future__ import annotations


def train_model(X_train, y_train):
    """Fit and return a classifier with predict_proba (XGBoost recommended).

    Handle the ~1.5% imbalance — e.g. xgboost's scale_pos_weight, or class
    weights / resampling.
    """
    raise NotImplementedError("Implement train_model")


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate on the held-out set.

    TODO: return at least {"roc_auc", "precision", "recall"} (add "pr_auc"
    and a confusion matrix in your write-up). Use predicted probabilities for
    roc_auc; pick a sensible threshold for precision/recall — NOT necessarily
    the default 0.5, since fraud is rare.
    """
    raise NotImplementedError("Implement evaluate")


def feature_importances(model, feature_names) -> list:
    """Return [(feature, importance)] sorted by importance, descending.

    Use feature_importances_ (trees/XGBoost) or coefficients (linear models).
    """
    raise NotImplementedError("Implement feature_importances")
