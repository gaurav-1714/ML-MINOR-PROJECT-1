# Minor Project 1: Breast Cancer Diagnosis Prediction

A supervised machine learning project that classifies breast tumors as **malignant** or **benign** using diagnostic measurements from fine needle aspirate (FNA) biopsy images.

---

## 1. Problem Statement

Breast cancer diagnosis traditionally relies on a pathologist visually examining biopsy samples, which can be time-consuming and subject to inter-observer variability. The objective of this project is to build a **supervised classification model** that predicts whether a breast tumor is **malignant (cancerous)** or **benign (non-cancerous)** based on quantitative measurements (radius, texture, perimeter, area, smoothness, concavity, etc.) extracted from digitized FNA images. Such a model could act as a decision-support tool, helping flag high-risk cases for closer clinical review.

**Type of problem:** Binary Classification (Supervised Learning)

---

## 2. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
- **Original Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
- **Kaggle Mirror:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Access used in this project:** `sklearn.datasets.load_breast_cancer()`, which bundles the identical dataset for easy, reproducible access. The exported copy used for this project is saved at [`data/breast_cancer.csv`](data/breast_cancer.csv).
- **Samples:** 569
- **Features:** 30 numeric features (mean, standard error, and "worst" value of 10 measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)
- **Target:** `diagnosis` — 0 = malignant, 1 = benign
- **Class balance:** 357 benign (62.7%) / 212 malignant (37.3%)

---

## 3. Data Preprocessing Steps

| Step | Details |
|---|---|
| Missing values | None found (0 missing values across all 569 × 30 features) |
| Duplicate rows | None found |
| Feature encoding | Target was already numerically encoded (0/1); no categorical features required encoding |
| Train/test split | 80% train (455 rows) / 20% test (114 rows), stratified by class |
| Feature scaling | `StandardScaler` (zero mean, unit variance), fitted on training data only and applied to both train and test sets to avoid data leakage |

---

## 4. Exploratory Data Analysis (EDA)

EDA artifacts are saved in [`results/`](results/):

| File | Description |
|---|---|
| `class_distribution.png` | Bar chart of malignant vs benign sample counts |
| `correlation_heatmap.png` | Correlation matrix of the "mean" feature group |
| `feature_distributions.png` | Histogram/KDE of 4 key features split by class |
| `feature_importance_rf.png` | Top-10 most predictive features per Random Forest |

**Key findings:**
- No missing values or duplicates — minimal cleaning required.
- Several size-related "mean" features (radius, perimeter, area) are highly correlated, as expected.
- Malignant tumors tend to show higher mean radius, texture, and concavity than benign tumors — a visibly separable pattern even before modeling.
- Concavity- and perimeter-related features are the strongest predictors of malignancy, consistent with the clinical understanding that irregular, concave tumor boundaries are a hallmark of malignancy.

---

## 5. Model(s) Used

Two supervised classification algorithms were trained and compared:

1. **Logistic Regression** — linear baseline, highly interpretable
2. **Random Forest Classifier** (200 trees) — non-linear ensemble model

Both were trained on the same scaled training data (`random_state=42` for reproducibility).

---

## 6. Training Process

1. Load dataset → `load_breast_cancer()`
2. Split into train (80%) / test (20%) sets, stratified by class
3. Fit `StandardScaler` on training features only; transform both sets
4. Train `LogisticRegression(max_iter=5000)` and `RandomForestClassifier(n_estimators=200)` on the scaled training data
5. Generate predictions and predicted probabilities on the held-out test set
6. Compute evaluation metrics and visualizations
7. Persist both trained models and the fitted scaler to `model/` using `joblib`

---

## 7. Evaluation Metrics and Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9932 |

**Confusion Matrices** (rows = actual, columns = predicted; labels = [malignant, benign]):

Logistic Regression:
```
[[41  1]
 [ 1 71]]
```

Random Forest:
```
[[39  3]
 [ 2 70]]
```

Supporting plots: `results/confusion_matrix_logistic_regression.png`, `results/confusion_matrix_random_forest.png`, `results/roc_curve_comparison.png`, `results/model_comparison.png`.

**Interpretation:**
- **Logistic Regression slightly outperformed Random Forest** on every metric for this dataset, achieving only 2 total misclassifications out of 114 test samples.
- Both models achieved ROC-AUC > 0.99, indicating excellent separability between the two classes.
- Random Forest made 2 false negatives (malignant predicted as benign) and 3 false positives, while Logistic Regression made only 1 of each — in a medical context, false negatives (missed cancers) are the more costly error type, making this an important metric to track beyond accuracy.

**Final model selected:** Logistic Regression (based on highest F1-score, strong interpretability, and lower error count).

---

## 8. Conclusion

This project demonstrates a complete supervised machine learning workflow — from problem framing and data exploration through preprocessing, model training, and rigorous evaluation — applied to a real-world medical diagnosis problem. Both candidate models performed strongly, but Logistic Regression was selected as the final model due to its slightly better metrics and superior interpretability, which is especially valuable in a healthcare decision-support context. The most informative features (concavity and perimeter measurements) align well with known clinical indicators of malignancy.

**Limitations & Future Work:**
- The dataset has moderate class imbalance (62.7%/37.3%); techniques like SMOTE or class-weighting could be explored.
- No hyperparameter tuning (e.g., `GridSearchCV`) was performed — this is a natural next step to further improve Random Forest performance.
- Cross-validation (e.g., 5-fold) would give a more robust estimate of generalization performance than a single train/test split.

---

## 9. References

1. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1995). *Breast Cancer Wisconsin (Diagnostic) Dataset*. UCI Machine Learning Repository.
2. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
3. Kaggle Dataset Mirror — https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

---

## Repository Structure

```
.
├── data/
│   └── breast_cancer.csv              # Exported dataset (569 rows x 32 cols)
├── model/
│   ├── logistic_regression_model.pkl  # Trained Logistic Regression model
│   ├── random_forest_model.pkl        # Trained Random Forest model
│   └── scaler.pkl                     # Fitted StandardScaler
├── notebook/
│   ├── Minor_Project_1_Breast_Cancer_Classification.ipynb   # Full annotated notebook
│   └── pipeline_script.py             # Same workflow as a plain .py script
├── results/
│   ├── class_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_distributions.png
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve_comparison.png
│   ├── model_comparison.png
│   ├── feature_importance_rf.png
│   ├── metrics_summary.json
│   └── metrics_summary.txt
└── README.md
```

## How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib jupyter

# Run as a script
python notebook/pipeline_script.py

# Or open the notebook
jupyter notebook notebook/Minor_Project_1_Breast_Cancer_Classification.ipynb
```

## Loading a Saved Model

```python
import joblib

model = joblib.load("model/logistic_regression_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# X_new must have the same 30 features, in the same order, as the training data
X_new_scaled = scaler.transform(X_new)
prediction = model.predict(X_new_scaled)
```
