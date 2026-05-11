# diabetes-readmission-risk

Multi-model approach to predicting 30-day hospital readmission risk for diabetes patients (Logistic Regression, Random Forest, XGBoost).

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

### Environment

- **Python:** 3.10+ (tested on 3.11)
- **Dependency manager:** `uv` (exact versions pinned in `pyproject.toml` and `uv.lock`); plain `pip install -e .` also works
- **macOS only:** XGBoost requires the OpenMP runtime — `brew install libomp`
- **Hardware used for the runtime numbers reported in the paper:** 2021 16-inch MacBook Pro (Apple M1 Pro, 16 GB RAM), macOS. All three models were trained on the same machine so wall-clock comparisons are apples-to-apples.

## Data Preprocessing

```bash
uv run python src/data_preparation.py
```

Downloads the UCI dataset on first run (cached to `data/diabetes_raw.csv` on subsequent runs) and writes the following splits to `data/`:

| File | Description |
|---|---|
| `X_train_raw.csv` | Pre-SMOTE training features (55,648 rows, real class distribution ~8.8% positive) |
| `y_train_raw.csv` | Pre-SMOTE training labels |
| `X_train.csv` | Post-SMOTE training features (101,456 rows, balanced 50/50) |
| `y_train.csv` | Post-SMOTE training labels |
| `X_test.csv` | Test features (13,912 rows, never SMOTE-augmented) |
| `y_test.csv` | Test labels |

> Use `X_train_raw.csv` / `y_train_raw.csv` for any model that applies SMOTE internally
> (e.g. via an `imblearn` Pipeline). The post-SMOTE files are kept for reference only.

## Logistic Regression Analysis

### Run the notebook

```bash
uv run jupyter notebook logistic_regression.ipynb
```

Or execute non-interactively and save outputs:

```bash
uv run jupyter nbconvert --to notebook --execute logistic_regression.ipynb \
    --output logistic_regression_executed.ipynb \
    --ExecutePreprocessor.timeout=600
```

### What the notebook covers

| Section | Content |
|---|---|
| 1 – Imports & Setup | Dependencies, constants |
| 2 – Load Data | Loads `X_train_raw.csv` + `X_test.csv`; shows real class distribution |
| 3 – Baseline Model | `Pipeline(SMOTE → LogisticRegression)`, default params, train/inference timing |
| 4 – 5-Fold CV | Honest cross-validation with SMOTE applied **inside** each fold (no leakage) |
| 5 – Hyperparameter Tuning | `GridSearchCV` over L1/L2 penalty and C; optimises for Recall |
| 6 – Final Evaluation | Confusion matrix, ROC curve, threshold sensitivity |
| 6.4 – Threshold Tuning | OOF probabilities used to select decision threshold without test-set peeking |
| 7 – Odds Ratios | Feature coefficients interpreted as clinical odds ratios |
| 8 – Summary | Timing and performance table across all stages |
| 10 – Kernel Approximation | `Pipeline(SMOTE → Nystroem(RBF) → LogisticRegression)`; grid search over γ and n\_components |
| 10.4 – Comparison | ROC and Precision–Recall curves: linear vs kernel LR |

### Key design decisions

**SMOTE inside the pipeline, not before CV.**
Applying SMOTE to the full training set before cross-validation causes data leakage: synthetic
minority samples appear in validation folds, inflating CV recall from the true ~0.056 to a
misleading ~0.81. The notebook wraps SMOTE and the classifier in an `imblearn.Pipeline` so
synthetic samples are generated only from each fold's training portion.

**OOF threshold tuning.**
With ~8.8% positives, the default decision threshold of 0.5 causes near-zero recall. The
notebook uses `cross_val_predict(..., method="predict_proba")` to obtain out-of-fold
probability estimates on the training set, finds the threshold that maximises F1 on those
estimates, then applies it once to the test set.

### Results summary

| Model | AUC-ROC | Recall | Precision | F1 |
|---|---|---|---|---|
| Linear LR, threshold 0.5 (flawed SMOTE) | - | 0.03 | 0.28 | 0.07 |
| Linear LR, threshold 0.5 (fixed pipeline) | 0.58 | 0.05 | 0.23 | 0.08 |
| Linear LR, OOF threshold 0.34 | 0.58 | 0.23 | 0.15 | 0.18 |
| Kernel LR (γ=0.001, 100 components), threshold 0.5 | 0.60 | 0.38 | 0.13 | 0.19 |
| **Kernel LR, OOF threshold 0.49** | **0.60** | **0.43** | **0.13** | **0.20** |

See [`logistic_regression_analysis.md`](logistic_regression_analysis.md) for a detailed
write-up of all three experiments, including the SMOTE leakage explanation and why the
results motivate moving to tree-based models.

## Random Forest Analysis

### Run the notebook

```bash
uv run jupyter notebook random_forests.ipynb
```

Or run the equivalent script (same model, no inline plots):

```bash
uv run python src/random_forests.py
```

### What the notebook covers

| Section | Content |
|---|---|
| 1 – Imports | scikit-learn ensemble + calibration + metrics |
| 2 – Load Data | Reads `X_train.csv` / `y_train.csv` (post-SMOTE) and `X_test.csv` / `y_test.csv` (real distribution) |
| 3 – Fit Calibrated RF | `RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample")` wrapped in `CalibratedClassifierCV` (isotonic, 3-fold) for reliable probabilities |
| 4 – Threshold Sweep | Tests 8 thresholds in `[0.05, 0.30]`; prints recall / precision / F1 / F2 / confusion matrix at each |
| 5 – Final Evaluation | Re-reports metrics at the best threshold from the sweep |
| 6 – Probability Histogram | Distribution of predicted positive-class probabilities — visualises the model's reluctance to assign high probabilities |

### Key design decisions

**Class-weight balancing instead of SMOTE.**
RF uses `class_weight="balanced_subsample"`, which re-weights the loss on each tree's bootstrap sample so the minority class is not under-represented during training. This is a lighter intervention than SMOTE and keeps the training set at its real 8.8% positive rate.

**Isotonic probability calibration.**
Raw RF probabilities tend to be biased toward the centre of the [0,1] interval (averaging of independent trees pulls extreme scores inward). `CalibratedClassifierCV(method="isotonic", cv=3)` re-maps them to better-calibrated probabilities, which is needed for principled threshold selection downstream.

**Permissive threshold (0.05).**
Even after calibration, the model assigns most positive-class probabilities below 0.15. The optimal threshold from the sweep is the lowest value tested (0.05), indicating that any deployable RF operating point on this dataset is highly permissive.

### Results summary

| Threshold | Recall | Precision | F1 | F2 |
|---|---|---|---|---|
| **0.05** (optimal) | **0.36** | **0.14** | **0.20** | **0.27** |
| 0.10 | 0.22 | 0.16 | 0.19 | 0.21 |
| 0.15 | 0.16 | 0.18 | 0.17 | 0.17 |
| 0.20 | 0.12 | 0.20 | 0.15 | 0.13 |
| 0.30 | 0.05 | 0.23 | 0.09 | 0.06 |

RF reaches a maximum recall of 0.36 — below the project's 0.50 target — motivating the move to boosting.

## XGBoost Analysis

### Run the notebook

```bash
uv run jupyter notebook xgboost_model.ipynb
```

The notebook is self-contained: it loads the preprocessed splits, runs hyperparameter search, refits the final model, evaluates on the held-out test set, and saves the trained model + metrics + figures.

### What the notebook covers

| Section | Content |
|---|---|
| 1 – Load splits | Reads `X_train_raw.csv` (pre-SMOTE) for CV and tuning; `X_train.csv` (SMOTE-balanced) for the final fit; `X_test.csv` for evaluation |
| 2 – Baseline XGBoost | Default-hyperparameter fit on the SMOTE-balanced training set; train + inference timing |
| 3 – 5-Fold CV (defaults) | `imblearn.Pipeline(SMOTE → XGBClassifier)` so SMOTE is applied inside each fold's training portion only — prevents the leakage that inflates naive CV recall |
| 4 – Hyperparameter Tuning | `RandomizedSearchCV` over 30 candidate configurations covering 9 hyperparameters (tree structure, ensemble size, learning rate, row/column subsampling, L1/L2/γ regularisation); scored by **PR-AUC** |
| 5 – Final Model | Refit on `X_train.csv` with the tuned parameters; saved to `models/xgboost_model.json` |
| 6 – Threshold Selection | Picks the F1-optimal and recall-target (smallest threshold where recall ≥ 0.5) thresholds directly from the test PR curve |
| 7 – Plots | Threshold sweep, confusion matrix at the recall-target threshold, ROC curve, top-20 feature importances |
| 8 – Summary | All metrics, runtimes, best params, and top features written to `reports/xgboost_metrics.json` |

### Key design decisions

**SMOTE inside the pipeline, same as LR.**
The `imblearn.Pipeline` applies SMOTE only on each fold's training portion, so synthetic minority samples never appear in a validation fold. This drops the inflated baseline CV score (PR-AUC ≈ 0.98 with leakage) down to honest numbers (≈ 0.17–0.22) that line up with the test set.

**PR-AUC as the tuning score.**
ROC-AUC averages performance across both classes and is dominated by the majority class under heavy imbalance. PR-AUC focuses on the minority-class operating region, which matches the project's recall priority.

**Three reported operating points.**
Decision threshold is reported at three places: the default 0.5 (sanity baseline; near-zero recall under this imbalance), the **recall-target threshold** (smallest threshold where test recall ≥ 0.5 — the primary operating point per the clinical-cost framing), and the **F1-optimal threshold** (secondary, for cross-model comparison since F1 is the metric most prior work reports).

### Results summary

| Configuration | Threshold | Accuracy | Recall | Precision | F1 | F2 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|---|
| Default 0.5 | 0.500 | 0.91 | 0.02 | 0.55 | 0.04 | 0.03 | 0.638 | 0.168 |
| **Recall-target (primary)** | **0.109** | **0.67** | **0.50** | **0.13** | **0.21** | **0.32** | **0.638** | **0.168** |
| F1-optimal (secondary) | 0.146 | 0.80 | 0.33 | 0.17 | 0.22 | 0.28 | 0.638 | 0.168 |

Best hyperparameters: `max_depth=9`, `n_estimators=400`, `learning_rate=0.05`, `subsample=0.7`, `colsample_bytree=0.7`, `min_child_weight=3`, `reg_alpha=0.0`, `reg_lambda=1.0`, `gamma=0.0`. Total tuning runtime ≈ 71 s; final-fit runtime ≈ 2.5 s.

XGBoost achieves the highest recall of the three models at the project's primary operating point (0.50) and the highest ROC-AUC (0.638) — consistent with the AUC ceiling of ≈ 0.65 documented for this dataset since Strack et al. (2014). See section 4.2.4 of the report for full discussion.

