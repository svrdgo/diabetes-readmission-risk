import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    recall_score,
    precision_score,
    fbeta_score
)

warnings.filterwarnings('ignore')



# def main():
#     X_train = pd.read_csv('data/X_train.csv')
#     y_train = pd.read_csv('data/y_train.csv').squeeze()

#     X_val = pd.read_csv("data/X_val.csv")
#     y_val = pd.read_csv("data/y_val.csv").squeeze()

#     X_test = pd.read_csv('data/X_test.csv')
#     y_test = pd.read_csv('data/y_test.csv').squeeze()


#     rf = RandomForestClassifier(
#         n_estimators=300,
#         random_state=42,
#         class_weight="balanced_subsample",
#         n_jobs=-1
#     )

#     model = CalibratedClassifierCV(rf, method='isotonic', cv=3)
#     model.fit(X_train, y_train)

#     y_probs = model.predict_proba(X_val)[:, 1]
#     tresholds =  [0.3, 0.4, 0.5, 0.6, 0.7]
#     for t in tresholds:
#         y_val_pred = (y_probs >= t).astype(int)

#         f2 = fbeta_score(y_val, y_val_pred, beta=2, zero_division=0)

#         print(f"\nValidation Threshold: {t}")

#         print("Recall:", recall_score(y_val, y_val_pred, zero_division=0))

#         print("Precision:", precision_score(y_val, y_val_pred, zero_division=0))

#         print("F2:", f2)

#         print(confusion_matrix(y_val, y_val_pred))
def main():
    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()

    X_val = pd.read_csv("data/X_val.csv")
    y_val = pd.read_csv("data/y_val.csv").squeeze()

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

    val_probs = model.predict_proba(X_val)[:, 1]
    thresholds = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3]

    best_t = None
    best_f2 = -1

    for t in thresholds:
        y_val_pred = (val_probs >= t).astype(int)
        f2 = fbeta_score(y_val, y_val_pred, beta=2, zero_division=0)

        print(f"\nValidation Threshold: {t}")
        print("Recall:", recall_score(y_val, y_val_pred, zero_division=0))
        print("Precision:", precision_score(y_val, y_val_pred, zero_division=0))
        print("F2:", f2)
        print(confusion_matrix(y_val, y_val_pred))

        if f2 > best_f2:
            best_f2 = f2
            best_t = t

    print(f"\nBest validation threshold: {best_t}")
    print(f"Best validation F2: {best_f2}")

    test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred = (test_probs >= best_t).astype(int)

    print("\nFinal Test Results")
    print("Recall:", recall_score(y_test, y_test_pred, zero_division=0))
    print("Precision:", precision_score(y_test, y_test_pred, zero_division=0))
    print("F2:", fbeta_score(y_test, y_test_pred, beta=2, zero_division=0))
    print(confusion_matrix(y_test, y_test_pred))

    print("\nTest probability summary:")
    print(pd.Series(test_probs).describe())

    plt.hist(test_probs, bins=50)
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title("Histogram of calibrated predicted probabilities")
    plt.show()

    # print("\nConfusion Matrix:")
    # print(confusion_matrix(y_test, y_pred))

   
    # print("\nClassification Report:")
    # print(classification_report(y_test, y_pred))

   
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # f2 = fbeta_score(y_test, y_pred, beta=2)

    # print("\nRecall :", recall)
    # print("Precision:", precision)
    # print("F2 Score:", f2)


 
    

    


if __name__ == "__main__":
    main()