# Disease Prediction using Machine Learning

A complete machine learning pipeline implemented for binary disease prediction using the **Breast Cancer Wisconsin Diagnostic dataset** from `scikit-learn`. This project demonstrates data preprocessing, supervised model training, evaluation metric collection, and visual performance reporting.

## Project Structure
- `disease_prediction.py`: The main Python script containing data loading, preprocessing, model training, evaluation, and plot generation.
- `results/`: Directory containing saved evaluation plots (`roc_curve_comparison_2.png`) and tabular metrics summary (`performance_metrics_2.csv`).
- `README.md`: Project documentation.

---

## Phase A: Data Preprocessing
1. **Dataset Loading**: Loaded features and binary target labels using `sklearn.datasets.load_breast_cancer`.
2. **Data Splitting**: Partitioned the dataset into training (80%) and testing (20%) sets using stratified sampling to maintain class distribution.
3. **Feature Scaling**: Applied `StandardScaler` to standardize all continuous features to zero mean and unit variance, ensuring stable convergence for distance-based and gradient-based algorithms.

---

## Phase B: Model Implementation
Three robust machine learning classifiers were trained and evaluated:
1. **Logistic Regression (LR)**: A linear baseline classifier suitable for binary outcomes.
2. **Random Forest (RF)**: An ensemble tree-based classifier that handles non-linear relationships and feature interactions.
3. **Support Vector Machine (SVM)**: A maximum-margin classifier using an RBF kernel to separate complex class boundaries.

---

## Phase C: Evaluation & Results
Models were evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. 

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 |
| **Random Forest** | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9939 |
| **Support Vector Machine** | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9950 |

### Visualizations
* **ROC Curves**: Saved under `results/roc_curve_comparison_2.png`, comparing the True Positive Rate vs. False Positive Rate across all models.
* **Metrics Export**: Detailed metrics saved under `results/performance_metrics_2.csv`.

---

## How to Run the Code
1. Ensure Python and required libraries are installed:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn
