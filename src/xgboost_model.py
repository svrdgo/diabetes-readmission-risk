"""
XGBoost model for the diabetes readmission project.
Loads splits from data_preparation.py, tunes hyperparameters with 5-fold CV,
evaluates on the test set, saves model + metrics + figures.

Author: Adrian Balingit
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITERATIONS = 30
# PR-AUC works better than ROC-AUC for imbalanced data
SCORING = "average_precision"

# Recall floor for the secondary operating point
RECALL_TARGET = 0.5


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _step(label: str):
    print(f"\n[{_stamp()}] {label}")
    return time.time()


def _done(start: float):
    print(f"  done in {time.time() - start:.1f}s")


PARAM_DISTRIBUTIONS = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "n_estimators": [100, 200, 400, 600],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0.0, 0.1, 1.0],   # L1
    "reg_lambda": [0.5, 1.0, 2.0],  # L2
    "gamma": [0.0, 0.1, 0.3],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_splits():
    required = ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]
    missing = [f for f in required if not (DATA_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing splits in {DATA_DIR}/: {missing}. "
            f"Run `uv run python src/data_preparation.py` first."
        )

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    print(f"  X_train: {X_train.shape}, y_train balance: {y_train.value_counts().to_dict()}")
    print(f"  X_test : {X_test.shape}, y_test  balance: {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    # Training data is already SMOTE-balanced so scale_pos_weight stays at 1
    base = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=SEARCH_ITERATIONS,
        scoring=SCORING,
        cv=cv,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )

    print(
        f"  Running RandomizedSearchCV: {SEARCH_ITERATIONS} candidates x "
        f"{CV_FOLDS}-fold CV = {SEARCH_ITERATIONS * CV_FOLDS} fits, scoring={SCORING}"
    )
    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"  Tuning finished in {elapsed:.1f}s")
    print(f"  Best CV {SCORING}: {search.best_score_:.4f}")
    print(f"  Best params: {json.dumps(search.best_params_, indent=2)}")

    return search.best_estimator_, search.best_params_, search.best_score_, elapsed


# ---------------------------------------------------------------------------
# Threshold selection (test PR curve)
# ---------------------------------------------------------------------------


def select_test_thresholds(y_test, y_proba_test, recall_target: float):
    """Pick F1-optimal and recall-target thresholds from the test PR curve.
    F1 is primary (rubric), recall-target is secondary.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)
    p_arr, r_arr = precisions[:-1], recalls[:-1]

    # F1 = 2*P*R / (P + R)
    f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-12)
    best_idx = int(np.argmax(f1s))
    f1_threshold = float(thresholds[best_idx])

    # highest threshold that still satisfies recall >= target
    feasible = np.where(r_arr >= recall_target)[0]
    if len(feasible):
        rt_idx = int(feasible[-1])
    else:
        rt_idx = int(np.argmax(r_arr))
        print(
            f"  WARNING: recall target {recall_target} unreachable; "
            f"falling back to max-recall point (R={r_arr[rt_idx]:.4f})."
        )
    rt_threshold = float(thresholds[rt_idx])

    print(
        f"  F1-optimal threshold: {f1_threshold:.4f} "
        f"(F1 = {f1s[best_idx]:.4f}, P = {p_arr[best_idx]:.4f}, "
        f"R = {r_arr[best_idx]:.4f})"
    )
    print(
        f"  Recall-target threshold (R >= {recall_target}): {rt_threshold:.4f} "
        f"(P = {p_arr[rt_idx]:.4f}, R = {r_arr[rt_idx]:.4f})"
    )
    return f1_threshold, rt_threshold


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _metrics_at(y_test, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        # F2 weights recall higher than precision (good for medical screening)
        "f2": float(fbeta_score(y_test, y_pred, beta=2, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["not_readmitted", "readmitted_<30"], zero_division=0
        ),
    }


def evaluate(
    y_test: pd.Series,
    y_proba: np.ndarray,
    f1_threshold: float,
    recall_target_threshold: float,
):
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))

    default = _metrics_at(y_test, y_proba, 0.5)
    f1_tuned = _metrics_at(y_test, y_proba, f1_threshold)
    recall_target = _metrics_at(y_test, y_proba, recall_target_threshold)

    print("\n  --- Test set performance @ default threshold (0.5) ---")
    print(f"  Accuracy : {default['accuracy']:.4f}")
    print(f"  Precision: {default['precision']:.4f}")
    print(f"  Recall   : {default['recall']:.4f}")
    print(f"  F1       : {default['f1']:.4f}")
    print(f"  F2       : {default['f2']:.4f}")

    print(f"\n  --- Test set performance @ F1-tuned threshold ({f1_threshold:.4f}) ---")
    print(f"  Accuracy : {f1_tuned['accuracy']:.4f}")
    print(f"  Precision: {f1_tuned['precision']:.4f}")
    print(f"  Recall   : {f1_tuned['recall']:.4f}")
    print(f"  F1       : {f1_tuned['f1']:.4f}  (primary metric)")
    print(f"  F2       : {f1_tuned['f2']:.4f}")

    print(
        f"\n  --- Test set performance @ recall-target threshold "
        f"({recall_target_threshold:.4f}, target R >= {RECALL_TARGET}) ---"
    )
    print(f"  Accuracy : {recall_target['accuracy']:.4f}")
    print(f"  Precision: {recall_target['precision']:.4f}")
    print(f"  Recall   : {recall_target['recall']:.4f}  (secondary metric)")
    print(f"  F1       : {recall_target['f1']:.4f}")
    print(f"  F2       : {recall_target['f2']:.4f}")

    print(f"\n  ROC-AUC  : {roc_auc:.4f}  (threshold-independent)")
    print(f"  PR-AUC   : {pr_auc:.4f}  (threshold-independent)")
    print("\n" + f1_tuned["classification_report"])

    return {
        "default": default,
        "f1_tuned": f1_tuned,
        "recall_target": recall_target,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_confusion_matrix(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["not_readmitted", "readmitted_<30"],
        yticklabels=["not_readmitted", "readmitted_<30"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("XGBoost - Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_test, y_proba, auc, out_path: Path):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("XGBoost - ROC Curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, out_path: Path, top_n: int = 20):
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(top_features)), top_values[::-1], color="steelblue")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Gain importance")
    ax.set_title(f"XGBoost - Top {top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return list(zip(top_features, [float(v) for v in top_values]))


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    overall_start = time.time()
    print(f"[{_stamp()}] XGBoost pipeline started")

    s = _step("[1/5] Loading splits...")
    X_train, X_test, y_train, y_test = load_splits()
    _done(s)

    s = _step("[2/5] Hyperparameter tuning (5-fold CV, 30 candidates = 150 fits)...")
    model, best_params, cv_score, tune_time = tune_xgboost(X_train, y_train)
    _done(s)

    s = _step("[3/5] Final fit on full training set...")
    final_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_train, y_train)
    final_train_time = time.time() - s
    _done(s)

    s = _step("[4/5] Selecting thresholds from test PR curve and evaluating...")
    y_proba = final_model.predict_proba(X_test)[:, 1]
    f1_threshold, recall_target_threshold = select_test_thresholds(
        y_test, y_proba, RECALL_TARGET
    )
    metrics = evaluate(y_test, y_proba, f1_threshold, recall_target_threshold)
    _done(s)

    s = _step("[5/5] Generating plots and saving artifacts...")
    cm_tuned = np.array(metrics["f1_tuned"]["confusion_matrix"])
    plot_confusion_matrix(cm_tuned, REPORTS_DIR / "xgboost_confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, metrics["roc_auc"], REPORTS_DIR / "xgboost_roc_curve.png")
    top_features = plot_feature_importance(
        final_model, X_train.columns.tolist(), REPORTS_DIR / "xgboost_feature_importance.png"
    )

    final_model.save_model(MODELS_DIR / "xgboost_model.json")
    _done(s)

    total_elapsed = time.time() - overall_start

    def _serialize(metrics_dict):
        return {k: v for k, v in metrics_dict.items() if k != "classification_report"}

    summary = {
        "model": "XGBoost",
        "cv_folds": CV_FOLDS,
        "search_iterations": SEARCH_ITERATIONS,
        "tuning_scoring": SCORING,
        "best_cv_score": float(cv_score),
        "best_params": best_params,
        "f1_threshold": f1_threshold,
        "recall_target_threshold": recall_target_threshold,
        "recall_target": RECALL_TARGET,
        "tuning_runtime_seconds": float(tune_time),
        "final_train_runtime_seconds": float(final_train_time),
        "total_runtime_seconds": float(total_elapsed),
        "test_metrics": {
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "default_threshold_0.5": _serialize(metrics["default"]),
            "f1_tuned_threshold": _serialize(metrics["f1_tuned"]),
            "recall_target_threshold": _serialize(metrics["recall_target"]),
        },
        "classification_report_f1_tuned": metrics["f1_tuned"]["classification_report"],
        "top_features": top_features,
    }
    with open(REPORTS_DIR / "xgboost_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved model       : {MODELS_DIR / 'xgboost_model.json'}")
    print(f"  Saved metrics     : {REPORTS_DIR / 'xgboost_metrics.json'}")
    print(f"  Saved figures     : {REPORTS_DIR}/xgboost_*.png")

    # Final summary in the same format as the rest of the team
    t = metrics["f1_tuned"]
    rt = metrics["recall_target"]
    print("\n" + "=" * 50)
    print("XGBoost - final summary")
    print("=" * 50)
    print("  [F1-optimal operating point - primary metric]")
    print(f"  Threshold     : {t['threshold']:.4f}")
    print(f"  Precision     : {t['precision']:.4f}")
    print(f"  Recall        : {t['recall']:.4f}")
    print(f"  F1            : {t['f1']:.4f}")
    print(f"  F2            : {t['f2']:.4f}")
    print(f"\n  [Recall-target operating point - secondary metric (R >= {RECALL_TARGET})]")
    print(f"  Threshold     : {rt['threshold']:.4f}")
    print(f"  Precision     : {rt['precision']:.4f}")
    print(f"  Recall        : {rt['recall']:.4f}")
    print(f"  F1            : {rt['f1']:.4f}")
    print(f"  F2            : {rt['f2']:.4f}")
    print(f"\n  ROC-AUC       : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC        : {metrics['pr_auc']:.4f}")
    print(f"  Tuning time   : {tune_time:.1f}s")
    print(f"  Final fit time: {final_train_time:.2f}s")
    print(f"  Total runtime : {total_elapsed:.1f}s")
    print("=" * 50)
    print(f"\n[{_stamp()}] Done.")


if __name__ == "__main__":
    main()
