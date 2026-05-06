# Logistic Regression Experiment — Analysis & Lessons Learned

**CS 549 · Early Intervention Through Prediction**  
Arshia Akhavan · Santiago Verdugo · Adrian Balingit

---

## 1. Problem Context

The goal is to predict whether a diabetes patient will be readmitted to hospital within 30 days
of discharge. This is a **binary classification** task with a heavily imbalanced target:
only ~8.8% of records are positive (early readmission).

The primary metric is **Recall** — in a clinical screening context, failing to identify a
high-risk patient (false negative) carries a higher cost than raising a false alarm (false
positive). A missed readmission may result in the patient not receiving preventive care,
leading to complications that are more expensive and harmful to treat later.

Accuracy alone is misleading here: a model that always predicts "not readmitted" achieves
91.2% accuracy while catching zero at-risk patients.

---

## 2. Dataset & Features

| Property | Value |
|---|---|
| Source | UCI ML Repository — Diabetes 130-US Hospitals (1999–2008) |
| Raw rows | 101,766 |
| After deduplication (first encounter per patient) | 71,518 |
| After dropping rows with missing `race` / `diag_1` | 69,560 |
| Train split (80%) | 55,648 |
| Test split (20%) | 13,912 |
| Positive rate (readmitted < 30 days) | 8.8% |

Features span four groups: demographics (race, gender, age), clinical encounter details
(time in hospital, admission type, discharge disposition, admission source), medical history
(lab/procedure/medication counts, prior visit counts), and diagnostics
(ICD-9 primary diagnosis grouped into 9 categories, A1C result, glucose serum result).

After one-hot encoding all categorical features the final feature matrix has 89 columns.
Numerical features were standard-scaled (fit on the training set only).

---

## 3. Experiment 1 — The Flawed Baseline

### 3.1 Setup

The first experiment applied SMOTE to the **entire training set** before any cross-validation
or model fitting:

```
Full training set (55,648 rows, 8.8% positive)
        │
        ▼
    SMOTE applied globally
        │
        ▼
Balanced training set (101,456 rows, 50% / 50%)
        │
        ├── saved as X_train.csv / y_train.csv
        │
        └── 5-fold CV and GridSearchCV run on this balanced dataset
```

A `LogisticRegression` (solver `lbfgs`, `max_iter=1000`) was trained and evaluated with
5-fold stratified cross-validation, then a grid search swept over penalty type (L1, L2)
and regularisation strength C ∈ {0.01, 0.1, 1.0, 10.0}.

### 3.2 Results

| Metric | CV mean (5-fold) | Test set |
|---|---|---|
| Accuracy | — | 0.91 |
| Precision (readmitted) | — | 0.28 |
| **Recall (readmitted)** | **0.81** | **0.03** |
| F1 (readmitted) | — | 0.07 |
| AUC-ROC | — | 0.60 |

The 5-fold CV reported an average Recall of **0.81**, which appeared excellent. However, when
the best model was evaluated on the held-out test set, Recall collapsed to **0.03**: the model
correctly identified only 41 out of 1,230 patients who were actually readmitted.

### 3.3 Root Cause — SMOTE Leakage

This catastrophic gap between CV recall and test recall has one root cause: **data leakage
through SMOTE**.

SMOTE (Synthetic Minority Over-sampling Technique) generates new synthetic minority-class
samples by interpolating between real minority samples in feature space. When SMOTE is applied
to the full training set *before* splitting into CV folds, those synthetic samples are
distributed across every fold. As a result:

- The **validation fold** contains synthetic samples that were generated from the same real
  minority samples used to train the model in that iteration.
- The model is effectively being tested on points that are close — in feature space — to points
  it was trained on.
- This makes the minority class appear much easier to predict than it really is.

Schematically:

```
WRONG (what we did):

Real minority samples → SMOTE → synthetic + real mixed together
                                        │
                     ┌──────────────────┴──────────────────────┐
                     │         5-fold split                     │
                     │                                          │
              ┌──────┴──────┐                           ┌───────┴──────┐
              │  Train fold │  ← real + synthetic       │  Val fold    │ ← real + synthetic
              └─────────────┘                           └──────────────┘
                                                              ▲
                                          validation contaminated with
                                          synthetic neighbours of train points


CORRECT (what we should do):

                     5-fold split on REAL data only
                     ┌──────────────────┬───────────────────────┐
                     │                  │                       │
              ┌──────┴──────┐           │               ┌───────┴──────┐
              │  Train fold │           │               │  Val fold    │ ← real only
              └──────┬──────┘           │               └──────────────┘
                     │                  │
                   SMOTE                │
                     │                  │
              augmented train           │
              (synthetic never seen by val fold)
```

The CV recall of 0.81 was therefore meaningless — it measured performance on synthetic
validation data, not on real unseen patients. The test set, containing only real patients,
exposed the true performance: recall of 0.03.

### 3.4 Secondary Issue — Wrong Decision Threshold

Even setting the leakage problem aside, Logistic Regression with the default threshold of 0.5
is poorly calibrated for a 9%-positive dataset.

The model outputs a probability estimate for each patient. At threshold 0.5, only predictions
with `P(readmitted) ≥ 0.5` are classified as positive. With such strong class imbalance, the
model learns that the base rate is low and most of its probability mass sits below 0.5 for the
minority class. On the test set, only 132 patients received a predicted probability above 0.5,
of whom 41 were true positives — hence recall of 0.03.

---

## 4. Experiment 2 — Fixing Both Problems

### 4.1 Fix 1: SMOTE Inside the CV Pipeline

The correct approach is to wrap SMOTE and the classifier in an `imblearn.pipeline.Pipeline`
and pass that pipeline — not pre-processed data — to `cross_validate` and `GridSearchCV`.
Scikit-learn's CV machinery calls `pipeline.fit(X_train_fold, y_train_fold)` on each fold,
which triggers SMOTE internally on that fold's training portion only. The validation fold
is never touched by SMOTE.

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("lr",    LogisticRegression(solver="saga", ...)),
])

# SMOTE is now applied inside each fold — no leakage
cross_validate(pipeline, X_train_raw, y_train_raw, cv=cv, ...)
```

We also saved `X_train_raw.csv` (pre-SMOTE, 55,648 rows) from the data preparation pipeline
so the notebook loads real-distribution training data.

### 4.2 Fix 2: OOF Threshold Tuning

The threshold of 0.5 is inappropriate for an imbalanced test set. Rather than picking a
threshold by looking at test-set results (which would constitute test-set peeking), we used
**out-of-fold (OOF) predictions**:

1. Run `cross_val_predict(..., method="predict_proba")` on the training set with the pipeline.
   This produces one probability estimate per training sample, each predicted by a model
   that never saw that sample.
2. Sweep thresholds and find the one that maximises F1 on the OOF training predictions.
3. Apply that threshold **once** to the test set.

This gives an unbiased threshold estimate and maintains a clean test-set evaluation.

### 4.3 Results After Both Fixes

#### CV results (honest, SMOTE per fold)

| | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Fold 1 | ~0.908 | ~0.310 | ~0.057 | ~0.096 | ~0.580 |
| Fold 2 | ~0.908 | ~0.325 | ~0.051 | ~0.086 | ~0.576 |
| Fold 3 | ~0.908 | ~0.340 | ~0.054 | ~0.091 | ~0.577 |
| Fold 4 | ~0.908 | ~0.330 | ~0.053 | ~0.089 | ~0.578 |
| Fold 5 | ~0.908 | ~0.330 | ~0.062 | ~0.103 | ~0.583 |
| **Mean** | **~0.908** | **~0.327** | **~0.055** | **~0.093** | **~0.579** |

CV Recall dropped from the inflated **0.81** to the honest **~0.056**, consistent with the
test-set recall.

#### Grid search

Best configuration: **L1 penalty, C = 0.01** (heavy regularisation).  
Best CV Recall: **0.056**

#### Test-set evaluation

| Configuration | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Default threshold (0.5) | 0.90 | 0.23 | 0.05 | 0.08 | 0.58 |
| OOF threshold (0.34) | 0.82 | 0.15 | **0.23** | 0.18 | 0.58 |

With the OOF threshold of 0.34, recall improves from 0.05 to **0.23** — the model now catches
roughly one in four readmissions instead of one in twenty.

---

## 5. Experiment 3 — Kernel Approximation (Nystroem + RBF)

### 5.1 Motivation

The linear model's AUC of 0.58 is near-random, confirming the data is not linearly separable.
A **kernel method** addresses this by implicitly mapping the original 89 features into a
higher-dimensional space φ(x) where a linear boundary can approximate a nonlinear one in the
original space. The RBF (Gaussian) kernel is:

```
K(x, x') = exp(−γ ‖x − x'‖²)
```

Rather than computing this kernel matrix exactly (O(n²) — infeasible at 55 k rows), we use
**Nystroem approximation**: sample `n_components` landmark points from the training data,
compute exact kernel evaluations against those landmarks, and use the resulting
`n_components`-dimensional embedding as explicit features fed to Logistic Regression.

The full pipeline remains: **SMOTE → Nystroem(RBF) → LogisticRegression**,
with SMOTE still applied inside each CV fold.

### 5.2 Hyperparameter Grid

| Parameter | Values searched |
|---|---|
| `kernel__gamma` | `None` (= 1/n\_features), 0.001, 0.01 |
| `kernel__n_components` | 100, 300 |
| `lr__C` | 0.1, 1.0 |

3 × 2 × 2 = 12 configurations × 5 folds = **60 fits** (wall time: ~61 s with `n_jobs=-1`).

### 5.3 Results

Best configuration found by grid search: **γ = 0.001, n\_components = 100, C = 0.1**

| Configuration | Accuracy | Precision | Recall | F1 | AUC-ROC | Train time |
|---|---|---|---|---|---|---|
| Kernel LR — default threshold (0.5) | 0.72 | 0.13 | 0.38 | 0.19 | **0.60** | 3.0 s |
| Kernel LR — OOF threshold (0.49) | 0.69 | 0.13 | **0.43** | 0.20 | 0.60 | — |

Compared to the best linear LR result:

| Model | AUC-ROC | Recall (best threshold) | F1 |
|---|---|---|---|
| Linear LR | 0.58 | 0.23 | 0.18 |
| **Kernel LR** | **0.60** | **0.43** | **0.20** |

The kernel approximation nearly **doubles recall** (0.23 → 0.43) and modestly improves
AUC-ROC (0.58 → 0.60). Training time also dropped dramatically — from 72 s (linear, large
SMOTE-balanced matrix) to **3 s** (Nystroem maps 55 k rows to 100 dimensions before SMOTE
expands the training set, making the LR solve much cheaper).

### 5.4 Why the Improvement Is Still Limited

Despite the gain, recall of 0.43 still means **57% of readmissions are missed** and
precision of 0.13 means **87% of flagged patients are false alarms**. Several reasons explain
the ceiling:

**Approximation quality is bounded by n_components.**  
Nystroem with 100 landmark points is a coarse approximation of the full RBF kernel space.
An exact kernel SVM on this data would produce a better boundary, but it would also require
O(n²) memory and O(n³) training time — impractical at 55 k rows.

**γ = 0.001 is very small.**  
A small γ means the kernel is very smooth and wide — effectively, each point influences a
large neighbourhood. With γ = 0.001 and 89 features, points that are quite different in
feature space are still considered similar. The model learns a nearly global decision boundary
that cannot resolve the local structure of the minority class. Larger γ values (0.01, 0.1)
scored lower in CV recall, suggesting the positive class is not well-localised in feature
space — there is no tight region of feature space that reliably predicts readmission.

**The positive class is intrinsically hard to separate.**  
The AUC-ROC of 0.60 after kernel mapping — compared to 0.58 for the linear model — tells us
that even a nonlinear Logistic Regression boundary provides only marginal additional separation.
This points to a **signal problem, not just a modelling problem**: the selected features may
not contain sufficient discriminative information to reliably predict 30-day readmission,
regardless of the classifier's capacity.

---

## 6. Why These Results Are Not Satisfactory

Even after kernel approximation, the best AUC-ROC reached is only **0.60** and the best
recall is **0.43** — still missing more than half of all readmissions. Several structural
reasons explain why no variant of Logistic Regression (linear or kernelised) is sufficient.

### 6.1 The Data Is Not Linearly Separable

Logistic Regression is a **linear classifier**. It partitions the feature space with a single
hyperplane. The decision boundary for readmission risk almost certainly does not take this form.
Readmission risk arises from complex, nonlinear interactions between clinical variables:

- A patient with many prior inpatient visits *and* a circulatory primary diagnosis *and*
  an abnormal A1C result may be at very high risk — but the combination matters, not any
  single feature independently.
- Age interacts with the number of medications, which interacts with discharge disposition.

Logistic Regression can only model these as additive, independent contributions. It cannot
capture feature interactions without explicit engineering of interaction terms.

### 6.2 The Minority Class Is Very Small and Heterogeneous

The positive class (readmission < 30 days) represents only 8.8% of patients. More
importantly, it is not a coherent, distinguishable group — it spans patients with very
different clinical profiles. SMOTE interpolates *within* the minority class, but if that
class is heterogeneous, synthetic samples may not occupy meaningful positions in feature
space, limiting how much they help.

### 6.3 Threshold Tuning Has a Hard Ceiling

Lowering the threshold increases recall but only by trading precision. At threshold 0.34:

- Recall = 0.23 → 77% of readmissions are **still missed**
- Precision = 0.15 → 85% of flagged patients are **false alarms**

The fundamental problem is that the AUC is low: no matter where we place the threshold, we
cannot simultaneously achieve high recall and acceptable precision when the model barely
discriminates between classes. Threshold tuning is a post-hoc adjustment; it cannot compensate
for a weak underlying model.

### 6.4 Recall vs Precision Trade-off at AUC = 0.58

For a random classifier, the precision-recall curve is a horizontal line at the positive rate
(8.8%). With AUC-ROC ≈ 0.58, the area under the precision-recall curve is only marginally
higher. This means:

- To achieve recall = 0.50 (catch half of readmissions), precision would fall to roughly
  the baseline rate (~10%), meaning the model provides almost no useful signal above random.
- Clinical utility requires a recall ≥ 0.60–0.70 with precision ≥ 0.20–0.25. Logistic
  Regression cannot reach this operating point on this dataset.

---

## 7. Summary of What We Learned

| Question | Finding |
|---|---|
| Was the original CV recall (0.81) real? | No. SMOTE leakage inflated it. True CV recall is ~0.056. |
| Does fixing the methodology improve test recall? | Yes, from 0.03 to 0.23 with OOF threshold tuning. |
| Does kernel approximation help? | Yes — recall improves from 0.23 to 0.43. AUC rises from 0.58 to 0.60. |
| Is 0.43 recall satisfactory? | No. 57% of readmissions still missed; precision only 0.13. |
| Is any form of Logistic Regression the right model here? | No. AUC ceiling ~0.60 even with nonlinear mapping. |
| What does this work give us? | A validated, leak-free pipeline and a performance floor for tree-based models. |

### Full Results Progression

| Model | AUC-ROC | Recall | Precision | F1 | Notes |
|---|---|---|---|---|---|
| Linear LR, threshold 0.5, SMOTE pre-CV | — | 0.03 | 0.28 | 0.07 | Flawed — leakage |
| Linear LR, threshold 0.5, honest pipeline | 0.58 | 0.05 | 0.23 | 0.08 | Fixed methodology |
| Linear LR, OOF threshold 0.34 | 0.58 | 0.23 | 0.15 | 0.18 | +threshold tuning |
| Kernel LR (γ=0.001, 100 components), threshold 0.5 | 0.60 | 0.38 | 0.13 | 0.19 | +nonlinear mapping |
| **Kernel LR, OOF threshold 0.49** | **0.60** | **0.43** | **0.13** | **0.20** | **Best result** |

### Lessons for the Next Models

1. **Keep SMOTE inside the pipeline** for all subsequent models (Random Forest, XGBoost).
2. **Use OOF threshold tuning** as standard practice.
3. **Do not trust CV metrics computed on pre-SMOTE data** — always verify that the metric
   distribution matches the real test-set distribution.
4. The AUC ceiling of ~0.60 for any logistic boundary strongly motivates tree-based models.
   **Random Forest** and **XGBoost** learn axis-aligned and interaction-based splits directly
   from data — no kernel approximation needed — and should achieve AUC well above 0.65–0.70
   if the signal exists.
5. The **odds ratios** from Logistic Regression (e.g., prior inpatient visits and discharge
   disposition strongly influencing readmission probability) remain useful as a sanity check
   on the feature importances that tree-based models will report.
