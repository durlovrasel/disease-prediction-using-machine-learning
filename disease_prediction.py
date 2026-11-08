"""
Disease Prediction using Machine Learning
Dataset: Breast Cancer Wisconsin Diagnostic (built into scikit-learn)

Run:
    pip install -r requirements.txt
    python disease_prediction.py

Positive class used for metric calculations: target 1 ("benign").
The dataset's target names and class encoding are supplied by scikit-learn.
This is an educational ML exercise, not a clinical diagnostic tool.
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 1. Load data
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target
print(f"Dataset shape: {X.shape}")
print("Target classes:", dict(enumerate(dataset.target_names)))
print("Missing values:", int(X.isna().sum().sum()))

# 2. Split before fitting transformations to avoid data leakage.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 3. Define three classification models.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
    ),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(probability=True, random_state=RANDOM_STATE))
    ])
}

# 4. Train, predict, and evaluate.
records = []
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    records.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities)
    })

    # Save one confusion matrix per model.
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=dataset.target_names, cmap="Blues", ax=ax
    )
    ax.set_title(f"Confusion Matrix — {name}")
    fig.tight_layout()
    filename = name.lower().replace(" ", "_")
    fig.savefig(RESULTS_DIR / f"confusion_matrix_{filename}.png", dpi=160)
    plt.close(fig)

# Save and print comparison table.
results = pd.DataFrame(records).sort_values("F1-score", ascending=False)
results.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)
print("\\nEvaluation metrics (test set):")
print(results.to_string(index=False))

# ROC curve comparison.
fig, ax = plt.subplots(figsize=(7, 5))
for name, model in models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.set_title("ROC Curves — Test Set")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "roc_curves.png", dpi=160)
plt.close(fig)

# Metric comparison plot.
chart = results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]]
ax = chart.plot(kind="bar", figsize=(10, 6), ylim=(0, 1.05))
ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=160)
plt.close()

print(f"\\nPlots and metrics saved in: {RESULTS_DIR.resolve()}")
