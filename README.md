# diabetes-readmission-risk

Multi-model approach to predicting 30-day hospital readmission risk for diabetes patients (Logistic Regression, Random Forest, XGBoost).

## Data Preprocessing

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
uv run python src/data_preparation.py
```

Downloads the UCI dataset automatically and writes cleaned train/test splits to `data/`.

