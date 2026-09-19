import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)


TRAINING_FILE = Path(
    "data/processed/ncr_district_day_training.csv"
)


print("=" * 60)
print("Baseline rainfall-only flood model")
print("=" * 60)


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(
    TRAINING_FILE,
    parse_dates=["date"]
)


# ------------------------------------------------------------
# 2. Define features
# ------------------------------------------------------------

features = [
    "rainfall_24h",
    "rainfall_3day",
    "rainfall_7day"
]

target = "flood_event"


# ------------------------------------------------------------
# 3. Temporal split
# ------------------------------------------------------------

train = df[
    df["date"] < "2022-01-01"
].copy()

test = df[
    df["date"] >= "2022-01-01"
].copy()


print()
print("Training period:")
print(
    train["date"].min(),
    "to",
    train["date"].max()
)

print(
    "Training rows:",
    len(train)
)

print(
    "Training flood days:",
    train[target].sum()
)


print()
print("Testing period:")
print(
    test["date"].min(),
    "to",
    test["date"].max()
)

print(
    "Testing rows:",
    len(test)
)

print(
    "Testing flood days:",
    test[target].sum()
)


# ------------------------------------------------------------
# 4. Prepare X and y
# ------------------------------------------------------------

X_train = train[features]

y_train = train[target]

X_test = test[features]

y_test = test[target]


# ------------------------------------------------------------
# 5. Train Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print()
print("Training Random Forest...")

model.fit(
    X_train,
    y_train
)


print("Training complete.")


# ------------------------------------------------------------
# 6. Predictions
# ------------------------------------------------------------

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ------------------------------------------------------------
# 7. Classification metrics
# ------------------------------------------------------------

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

pr_auc = average_precision_score(
    y_test,
    y_probability
)


print()
print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1 score:  {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)


# ------------------------------------------------------------
# 8. Confusion matrix
# ------------------------------------------------------------

print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ------------------------------------------------------------
# 9. Classification report
# ------------------------------------------------------------

print()
print("Classification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 10. Feature importance
# ------------------------------------------------------------

print()
print("Feature importance:")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(
    importance.to_string(index=False)
)


print()
print("=" * 60)
print("Baseline model complete.")
print("=" * 60)