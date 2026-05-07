# diabetes-readmission-risk

Multi-model approach to predicting 30-day hospital readmission risk for diabetes
patients. Track 2 of the CS 549 (Spring 2026) final project.

Source dataset: [Diabetes 130-US Hospitals for Years 1999-2008](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008)
(UCI ML Repository, ID 296). Binary target: `1` if readmitted within 30 days,
else `0`. Recall is the primary metric.

Team & assignments:
- Arshia Akhavan — Logistic Regression
- Santiago Verdugo — Random Forest
- Adrian Balingit — XGBoost (this README documents the XGBoost pipeline)

## Repository Layout

```
diabetes-readmission-risk/
├── pyproject.toml             # dependencies (uv-managed)
├── README.md                  # this file
├── src/
│   ├── data_preparation.py    # shared preprocessing → data/X_train.csv, etc.
│   └── xgboost_model.py       # XGBoost training + evaluation
├── data/                      # (gitignored) preprocessed splits
├── models/                    # (gitignored) trained model artifacts
└── reports/                   # (gitignored) metrics JSON + figures
```

All three team models consume the same splits written by
`src/data_preparation.py` so the comparison is fair (same features, same
SMOTE-balanced training set, same scaling, same test split).

## Environment

- **Python**: 3.10+ (tested on 3.11)
- **Dependency manager**: [uv](https://docs.astral.sh/uv/) (also runs fine with
  plain `pip` — see Colab instructions below)
- **OS**: macOS / Linux / Windows. On macOS, XGBoost requires the OpenMP
  runtime (`brew install libomp`).

Key libraries (pinned via `pyproject.toml`):

| Package           | Purpose                                          |
|-------------------|--------------------------------------------------|
| `ucimlrepo`       | Auto-downloads the UCI dataset                   |
| `pandas`, `numpy` | Data manipulation                                |
| `scikit-learn`    | Splitting, scaling, CV, metrics                  |
| `imbalanced-learn`| SMOTE oversampling                               |
| `xgboost`         | Gradient-boosted trees model                     |
| `matplotlib`, `seaborn` | Confusion matrix / ROC / feature importance plots |

## How to Run — Local Machine

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh   # macOS / Linux
# (or:  winget install --id=astral-sh.uv  on Windows)

# 2. macOS only: install OpenMP runtime for XGBoost
brew install libomp

# 3. Sync dependencies (creates .venv automatically)
cd diabetes-readmission-risk
uv sync

# 4. Run the shared preprocessing pipeline
uv run python src/data_preparation.py

# 5. Train and evaluate XGBoost
uv run python src/xgboost_model.py
```

Step 4 downloads the UCI dataset, deduplicates patients, handles missing
values, ICD-9-groups `diag_1`, encodes categoricals, splits 80/20, scales
numerical features, SMOTE-balances training only, and writes the splits to
`data/`. Step 5 produces the artifacts in `models/` and `reports/`.

Using plain `pip` instead of `uv`:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # reads pyproject.toml
python src/data_preparation.py
python src/xgboost_model.py
```

## How to Run — Google Colab

Colab provides Linux + OpenMP out of the box, so no `libomp` install is
needed. A full run (preprocessing + tuning + evaluation) takes about 1–2
minutes on a free CPU runtime.

### Option A — Use teammate's preprocessed CSVs (recommended)

If a teammate has already run `data_preparation.py` and shared the four
output files (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`),
this is the cleanest path — the model trains on the exact same bytes
they used, so cross-model comparison is identical.

```python
# Cell 1 — upload xgboost_model.py and the four preprocessed CSVs
from google.colab import files
import os
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)

uploaded = files.upload()   # select: xgboost_model.py + X_train.csv + X_test.csv + y_train.csv + y_test.csv
for name in uploaded:
    if name.endswith(".py"):
        os.rename(name, f"src/{name}")
    elif name.endswith(".csv"):
        os.rename(name, f"data/{name}")
```

```python
# Cell 2 — install dependencies (no ucimlrepo needed since we skip the download)
!pip install -q pandas numpy scikit-learn xgboost matplotlib seaborn
```

```python
# Cell 3 — train and evaluate XGBoost
!python src/xgboost_model.py
```

```python
# Cell 4 — view results inline
from IPython.display import Image, display
import json
print(json.dumps(json.load(open("reports/xgboost_metrics.json")), indent=2))
display(Image("reports/xgboost_confusion_matrix.png"))
display(Image("reports/xgboost_roc_curve.png"))
display(Image("reports/xgboost_feature_importance.png"))
```

### Option B — Run preprocessing yourself (downloads from UCI)

Use this if no teammate has shared their CSVs yet. The preprocessing is
deterministic (`RANDOM_STATE = 42`), so the output is identical across
runs anyway.

```python
# Cell 1 — upload both .py files
from google.colab import files
import os
os.makedirs("src", exist_ok=True)
uploaded = files.upload()                # pick data_preparation.py + xgboost_model.py
for name in uploaded:
    os.rename(name, f"src/{name}")
```

```python
# Cell 2 — install deps (ucimlrepo needed for the UCI download)
!pip install -q ucimlrepo pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
```

```python
# Cell 3 — preprocessing (auto-downloads UCI data, ~10 seconds)
!python src/data_preparation.py
```

```python
# Cell 4 — train and evaluate XGBoost
!python src/xgboost_model.py
```

```python
# Cell 5 — view results inline
from IPython.display import Image, display
import json
print(json.dumps(json.load(open("reports/xgboost_metrics.json")), indent=2))
display(Image("reports/xgboost_confusion_matrix.png"))
display(Image("reports/xgboost_roc_curve.png"))
display(Image("reports/xgboost_feature_importance.png"))
```

### Option C — Upload the whole folder as a zip

Locally: zip the project folder (e.g. right-click → "Compress" on macOS, or
`zip -r diabetes-readmission-risk.zip diabetes-readmission-risk/`). Then in Colab:

```python
from google.colab import files
uploaded = files.upload()                # pick diabetes-readmission-risk.zip
!unzip -q diabetes-readmission-risk.zip
%cd diabetes-readmission-risk
!pip install -q ucimlrepo pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
!python src/data_preparation.py
!python src/xgboost_model.py
```

### Option D — Mount Google Drive (if the folder is in your Drive)

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/path/to/diabetes-readmission-risk
!pip install -q ucimlrepo pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
!python src/data_preparation.py
!python src/xgboost_model.py
```

### Option E — With Git (after the repo is pushed)

```python
!git clone https://github.com/<your-team>/diabetes-readmission-risk.git
%cd diabetes-readmission-risk
!pip install -q ucimlrepo pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn
!python src/data_preparation.py
!python src/xgboost_model.py
```

## XGBoost Methodology (what the model does)

### Inputs

`src/xgboost_model.py` consumes the four CSV files written by
`src/data_preparation.py`:

- `data/X_train.csv` — features, SMOTE-balanced (~50/50 class split)
- `data/y_train.csv` — labels for `X_train`
- `data/X_test.csv` — features, original imbalanced distribution (~91/9)
- `data/y_test.csv` — labels for `X_test`

It does **not** redo any preprocessing — that's intentional so the three
team models train and evaluate on identical data.

### Training

A `xgboost.XGBClassifier` (objective `binary:logistic`, `tree_method="hist"`)
is fit with hyperparameters chosen by `RandomizedSearchCV`:

- 5-fold stratified cross-validation
- 30 random parameter combinations (150 fits total)
- Scoring: **ROC-AUC** (threshold-independent — picks the model with the
  best discrimination irrespective of decision cutoff)

The search space covers every hyperparameter listed in the project proposal
plus L1/L2 regularization:

| Parameter           | Values                            |
|---------------------|-----------------------------------|
| `learning_rate`     | 0.01, 0.05, 0.1, 0.2              |
| `max_depth`         | 3, 5, 7, 9                        |
| `n_estimators`      | 100, 200, 400, 600                |
| `subsample`         | 0.7, 0.85, 1.0                    |
| `colsample_bytree`  | 0.7, 0.85, 1.0                    |
| `min_child_weight`  | 1, 3, 5                           |
| `gamma`             | 0.0, 0.1, 0.3                     |
| `reg_alpha` (L1)    | 0.0, 0.1, 1.0                     |
| `reg_lambda` (L2)   | 0.5, 1.0, 2.0                     |

Because the training set is already SMOTE-balanced, `scale_pos_weight=1`.

### Decision threshold (Bayes prior correction)

Training on SMOTE-balanced data calibrates the model's predicted
probabilities to a 50/50 prior, but the test set has a ~9/91 prior. A naive
0.5 cutoff yields high accuracy and near-zero recall — clinically useless.

The script applies a **Bayes prior shift** to recalibrate every predicted
probability from the SMOTE training prior to the population prior, then
selects the F1-optimal decision threshold using **out-of-fold cross-validated
probabilities** (no test-set leakage). Test metrics are reported at both
the default 0.5 cutoff and the tuned cutoff for transparency.

### Evaluation metrics

Per the proposal, recall is the primary metric (false negatives are clinically
costly). The script also reports:

- **ROC-AUC** and **PR-AUC** — threshold-independent, ideal for cross-model comparison
- **Recall**, **Precision**, **F1**, **Accuracy** — at both thresholds
- **Confusion matrix** — at the tuned threshold
- **Per-class classification report**
- **Tuning runtime** and **final-fit runtime** — runtime analysis required by the proposal

### Outputs

After running `src/xgboost_model.py`:

| File                                     | Contents                                       |
|------------------------------------------|------------------------------------------------|
| `models/xgboost_model.json`              | Trained XGBoost model (loadable via `xgb.XGBClassifier().load_model(...)`) |
| `reports/xgboost_metrics.json`           | Best params, all test metrics, top-20 features, runtimes, prior info |
| `reports/xgboost_confusion_matrix.png`   | Test-set confusion matrix at tuned threshold   |
| `reports/xgboost_roc_curve.png`          | Test-set ROC curve with AUC                    |
| `reports/xgboost_feature_importance.png` | Top 20 features by gain importance             |

## Reproducibility

`RANDOM_STATE = 42` is used for the train/test split, SMOTE, the K-fold CV,
and the XGBoost classifier itself, so reruns produce identical results given
the same package versions.
