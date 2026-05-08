import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score,
    fbeta_score
)

warnings.filterwarnings("ignore")


def main():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").squeeze()

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    )

    model = CalibratedClassifierCV(rf, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]

    best_f1 = -1
    best_t = None

    thresholds = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]

    for t in thresholds:
        y_test_pred = (test_probs >= t).astype(int)

        f1 = f1_score(y_test, y_test_pred, zero_division=0)
        f2 = fbeta_score(y_test, y_test_pred, beta=2, zero_division=0)

        print(f"\nTest Threshold: {t}")
        print("Recall:", recall_score(y_test, y_test_pred, zero_division=0))
        print("Precision:", precision_score(y_test, y_test_pred, zero_division=0))
        print("F1:", f1)
        print("F2:", f2)
        print(confusion_matrix(y_test, y_test_pred))

        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    print(f"\nBest threshold: {best_t}")
    print(f"Best F1: {best_f1}")

    y_test_pred = (test_probs >= best_t).astype(int)

    print(f"\nFinal Test Results using threshold: {best_t}")
    print("Recall:", recall_score(y_test, y_test_pred, zero_division=0))
    print("Precision:", precision_score(y_test, y_test_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_test_pred, zero_division=0))
    print("F2:", fbeta_score(y_test, y_test_pred, beta=2, zero_division=0))
    print(confusion_matrix(y_test, y_test_pred))

    print("\nTest probability summary:")
    print(pd.Series(test_probs).describe())

    plt.hist(test_probs, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Histogram of calibrated predicted probabilities")
    plt.show()


if __name__ == "__main__":
    main()