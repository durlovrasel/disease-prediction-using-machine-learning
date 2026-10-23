# Disease Prediction using Machine Learning

This repository contains a complete machine learning pipeline for binary disease prediction using the **Breast Cancer Wisconsin Diagnostic dataset** from `scikit-learn`.

## Project Structure
- `notebook.ipynb` (or `script.py`): Main code containing data preprocessing, model training, and evaluation.
- `results/`: Directory containing saved evaluation plots and metrics summary CSV.
- `README.md`: Project documentation.

## Phase A: Data Preprocessing
1. **Dataset Loading**: Loaded features and target labels from `sklearn.datasets.load_breast_cancer`.
2. **Data Splitting**: Split the dataset into 80% training and 20% testing sets using stratified sampling.
3. **Feature Scaling**: Applied `StandardScaler` to normalize all continuous features to zero mean and unit variance.

## Phase B: Model Implementation
Three supervised machine learning algorithms were implemented and trained:
1. **Logistic Regression (LR)**: Linear baseline classifier.
2. **Random Forest (RF)**: Ensemble tree-based classifier.
3. **Support Vector Machine (SVM)**: Maximum-margin hyperplane classifier with an RBF kernel.

## Phase C: Evaluation & Results
Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. 

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | ~0.97 | ~0.96 | ~0.99 | ~0.97 | ~0.99 |
| Random Forest | ~0.96 | ~0.95 | ~0.99 | ~0.96 | ~0.99 |
| Support Vector Machine | ~0.97 | ~0.96 | ~0.99 | ~0.97 | ~0.99 |

### Saved Visualizations
* `results/roc_curve_comparison.png`: Receiver Operating Characteristic curves comparing all models.
* `results/performance_metrics.csv`: Tabular summary of metrics.