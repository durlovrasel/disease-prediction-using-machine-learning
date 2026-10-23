import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Create results directory for plots
os.makedirs("results", exist_ok=True)

# ==========================================
# PHASE A: DATA PREPROCESSING
# ==========================================
print("--- Phase A: Loading and Preprocessing Data ---")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set shape: {X_train_scaled.shape}")
print(f"Testing set shape: {X_test_scaled.shape}")

# ==========================================
# PHASE B: MODEL IMPLEMENTATION
# ==========================================
print("\n--- Phase B: Training Models ---")
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
}

trained_models = {}
for name, model in models.items():
  model.fit(X_train_scaled, y_train)
  trained_models[name] = model
  print(f"Trained {name} successfully.")

# ==========================================
# PHASE C: EVALUATION & METRICS
# ==========================================
print("\n--- Phase C: Evaluating Models ---")
evaluation_results = []

plt.figure(figsize=(8, 6))

for name, model in trained_models.items():
  y_pred = model.predict(X_test_scaled)
  y_prob = model.predict_proba(X_test_scaled)[:, 1]

  acc = accuracy_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  rec = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_prob)

  evaluation_results.append({
      "Model": name,
      "Accuracy": acc,
      "Precision": prec,
      "Recall": rec,
      "F1-Score": f1,
      "ROC-AUC": roc_auc,
  })

  # Plot ROC Curve
  fpr, tpr, _ = roc_curve(y_test, y_prob)
  plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

# Finalize and save ROC curve plot
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.savefig("results/roc_curve_comparison.png")
plt.close()

# Convert results to DataFrame and display
results_df = pd.DataFrame(evaluation_results)
print("\nPerformance Comparison:")
print(results_df.to_string(index=False))

# Save results summary to CSV
results_df.to_csv("results/performance_metrics.csv", index=False)
print("\nPlots and metrics successfully saved in the 'results/' folder!")