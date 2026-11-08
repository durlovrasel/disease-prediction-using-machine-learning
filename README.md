# Disease Prediction using Machine Learning

## Project overview
This project demonstrates a reproducible machine-learning classification workflow using the **Breast Cancer Wisconsin Diagnostic dataset** included in scikit-learn. It compares Logistic Regression, Random Forest, and Support Vector Machine (SVM).

> **Important:** This is an educational assignment, not a clinical diagnostic tool. It must not be used for patient diagnosis or treatment decisions.

## Dataset
- Source: `sklearn.datasets.load_breast_cancer` (scikit-learn built-in dataset).
- Features: 30 numeric measurements.
- Samples: 569.
- Target classes: malignant and benign.
- Target encoding from scikit-learn: `0 = malignant`, `1 = benign`.
- Dataset documentation: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

## Preprocessing
1. Load the dataset directly from scikit-learn.
2. Inspect dimensions, target names, and missing values.
3. Split data into 80% training and 20% test data using stratification and `random_state=42`.
4. Standardize features for Logistic Regression and SVM using a pipeline. The scaler is fitted on training data only.
5. Random Forest is trained without scaling.

## Models
- Logistic Regression
- Random Forest (300 trees)
- Support Vector Machine (RBF kernel, scikit-learn default)

## Evaluation
Metrics are calculated on the held-out test set. Scikit-learn encodes benign as class `1`, so precision, recall, and F1-score use benign as the positive class.

| Model                  |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |
|:-----------------------|-----------:|------------:|---------:|-----------:|----------:|
| Logistic Regression    |     0.9825 |      0.9861 |   0.9861 |     0.9861 |    0.9954 |
| Support Vector Machine |     0.9825 |      0.9861 |   0.9861 |     0.9861 |    0.995  |
| Random Forest          |     0.9474 |      0.9583 |   0.9583 |     0.9583 |    0.9937 |

These are one reproducible split's results, not a clinical performance guarantee. For a medical context, also examine malignant-class sensitivity/recall, false negatives, calibration, external validation, and clinical oversight.

## Results figures
The `results/` directory contains:
- `model_metrics.csv` — metric comparison table
- `model_comparison.png` — bar chart comparing metrics
- `roc_curves.png` — ROC curves
- `confusion_matrix_*.png` — confusion matrix for each model

## Repository files
- `disease_prediction.ipynb` — commented notebook with preprocessing, training, evaluation, and plots
- `disease_prediction.py` — equivalent runnable Python script
- `requirements.txt` — dependencies
- `results/` — generated figures and metrics

## How to run
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python disease_prediction.py
```
Or open `disease_prediction.ipynb` in Jupyter Notebook/JupyterLab and run all cells.

## Limitations
- A single random train/test split is used; cross-validation and external validation would provide stronger evidence.
- The dataset is a benchmark dataset and may not represent current clinical populations or workflows.
- The positive class for the main metric table is benign (label 1). Always inspect class-specific confusion matrices and malignant recall before interpreting performance.
- This project is for education only, not medical advice or diagnosis.

## Submission
1. Create a GitHub repository and upload the files in this project.
2. Copy the repository's public URL.
3. Paste that URL into the assignment spreadsheet: https://docs.google.com/spreadsheets/d/1TE-GX5WuaHl4q-ExEnAV-Axb5YknRXWnA4q l4kHCBuI/edit?usp=sharing

The repository URL and spreadsheet submission are **not** completed automatically by these files; you must publish the repository and enter its URL.
