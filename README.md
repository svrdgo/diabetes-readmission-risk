# diabetes-readmission-risk

Multi-model approach to predicting 30-day hospital readmission risk for diabetes patients (Logistic Regression, Random Forest, XGBoost).

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

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
| Linear LR, threshold 0.5 (flawed SMOTE) | — | 0.03 | 0.28 | 0.07 |
| Linear LR, threshold 0.5 (fixed pipeline) | 0.58 | 0.05 | 0.23 | 0.08 |
| Linear LR, OOF threshold 0.34 | 0.58 | 0.23 | 0.15 | 0.18 |
| Kernel LR (γ=0.001, 100 components), threshold 0.5 | 0.60 | 0.38 | 0.13 | 0.19 |
| **Kernel LR, OOF threshold 0.49** | **0.60** | **0.43** | **0.13** | **0.20** |

See [`logistic_regression_analysis.md`](logistic_regression_analysis.md) for a detailed
write-up of all three experiments, including the SMOTE leakage explanation and why the
results motivate moving to tree-based models.

