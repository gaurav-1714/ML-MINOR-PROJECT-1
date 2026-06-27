"""
Minor Project 1 - Breast Cancer Diagnosis Prediction
Supervised Machine Learning Pipeline

This script performs the complete ML workflow:
1. Data loading
2. Exploratory Data Analysis (EDA)
3. Data preprocessing
4. Model training (Logistic Regression & Random Forest)
5. Model evaluation
6. Saving all artifacts (data, models, plots, metrics)
"""

import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

DATA_DIR = "data"
MODEL_DIR = "model"
RESULTS_DIR = "results"

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
print("### SECTION 1: LOAD DATA ###")
bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()
df.rename(columns={"target": "diagnosis"}, inplace=True)
# target: 0 = malignant, 1 = benign (sklearn encoding) -> we map to readable labels for EDA
df["diagnosis_label"] = df["diagnosis"].map({0: "malignant", 1: "benign"})

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().to_string())

df.to_csv(f"{DATA_DIR}/breast_cancer.csv", index=False)
print(f"\nSaved raw dataset to {DATA_DIR}/breast_cancer.csv")

# ----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ----------------------------------------------------------------------
print("\n### SECTION 2: EXPLORATORY DATA ANALYSIS ###")

print("\nMissing values per column (top 5):")
print(df.isnull().sum().sort_values(ascending=False).head())
print("\nTotal missing values:", int(df.isnull().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

print("\nClass distribution:")
print(df["diagnosis_label"].value_counts())

print("\nStatistical summary (selected features):")
key_features = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness"]
print(df[key_features].describe().to_string())

# -- Plot 1: Class distribution
plt.figure(figsize=(6, 4))
ax = sns.countplot(data=df, x="diagnosis_label", palette=["#e74c3c", "#2ecc71"])
ax.set_title("Class Distribution: Malignant vs Benign")
ax.set_xlabel("Diagnosis")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=10)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/class_distribution.png", dpi=150)
plt.close()
print(f"\nSaved {RESULTS_DIR}/class_distribution.png")

# -- Plot 2: Correlation heatmap (mean features only, for readability)
mean_cols = [c for c in df.columns if c.startswith("mean ")]
plt.figure(figsize=(10, 8))
corr = df[mean_cols].corr()
sns.heatmap(corr, cmap="coolwarm", annot=False, square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap (Mean Features)")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/correlation_heatmap.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/correlation_heatmap.png")

# -- Plot 3: Feature distributions by class
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
dist_feats = ["mean radius", "mean texture", "mean concavity", "mean smoothness"]
for ax, feat in zip(axes.flatten(), dist_feats):
    sns.histplot(data=df, x=feat, hue="diagnosis_label", kde=True, ax=ax,
                 palette=["#e74c3c", "#2ecc71"], alpha=0.6)
    ax.set_title(f"Distribution of {feat}")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_distributions.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/feature_distributions.png")

# ----------------------------------------------------------------------
# 3. DATA PREPROCESSING
# ----------------------------------------------------------------------
print("\n### SECTION 3: DATA PREPROCESSING ###")

X = df[bc.feature_names.tolist()]
y = df["diagnosis"]  # 0 = malignant, 1 = benign

print("No missing values and no duplicates found, so no imputation/row removal was required.")
print("Feature encoding: target is already numeric (0 = malignant, 1 = benign); no categorical features present.")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain set size: {X_train.shape[0]} rows")
print(f"Test set size: {X_test.shape[0]} rows")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Applied StandardScaler (mean=0, std=1) feature scaling, fit on training data only.")

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
print(f"Saved fitted scaler to {MODEL_DIR}/scaler.pkl")

# ----------------------------------------------------------------------
# 4. MODEL DEVELOPMENT
# ----------------------------------------------------------------------
print("\n### SECTION 4: MODEL DEVELOPMENT ###")

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}

trained_models = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model
    print(f"Trained: {name}")

# ----------------------------------------------------------------------
# 5. MODEL EVALUATION
# ----------------------------------------------------------------------
print("\n### SECTION 5: MODEL EVALUATION ###")

results = {}
plt.figure(figsize=(7, 6))

for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "confusion_matrix": cm.tolist(),
    }

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Individual confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["malignant", "benign"])
    disp.plot(ax=ax_cm, cmap="Blues", colorbar=False)
    ax_cm.set_title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{fname}.png", dpi=150)
    plt.close(fig_cm)
    print(f"Saved {RESULTS_DIR}/confusion_matrix_{fname}.png")

    # Add to combined ROC plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/roc_curve_comparison.png", dpi=150)
plt.close()
print(f"\nSaved {RESULTS_DIR}/roc_curve_comparison.png")

# -- Model comparison bar chart
comp_df = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1_score"]]
ax = comp_df.plot(kind="bar", figsize=(8, 5), rot=0, colormap="viridis")
ax.set_title("Model Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0.8, 1.0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/model_comparison.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/model_comparison.png")

# -- Feature importance (Random Forest)
rf = trained_models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette="mako")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/feature_importance_rf.png", dpi=150)
plt.close()
print(f"Saved {RESULTS_DIR}/feature_importance_rf.png")

# ----------------------------------------------------------------------
# 6. SAVE MODELS & METRICS
# ----------------------------------------------------------------------
print("\n### SECTION 6: SAVE MODELS & METRICS ###")

for name, model in trained_models.items():
    fname = name.lower().replace(" ", "_")
    joblib.dump(model, f"{MODEL_DIR}/{fname}_model.pkl")
    print(f"Saved {MODEL_DIR}/{fname}_model.pkl")

best_model_name = max(results, key=lambda k: results[k]["f1_score"])
print(f"\nBest performing model based on F1-score: {best_model_name}")

with open(f"{RESULTS_DIR}/metrics_summary.json", "w") as f:
    json.dump({"results": results, "best_model": best_model_name}, f, indent=2)
print(f"Saved {RESULTS_DIR}/metrics_summary.json")

with open(f"{RESULTS_DIR}/metrics_summary.txt", "w") as f:
    f.write("MODEL EVALUATION SUMMARY\n")
    f.write("=" * 40 + "\n\n")
    for name, m in results.items():
        f.write(f"{name}\n")
        f.write(f"  Accuracy : {m['accuracy']}\n")
        f.write(f"  Precision: {m['precision']}\n")
        f.write(f"  Recall   : {m['recall']}\n")
        f.write(f"  F1-score : {m['f1_score']}\n")
        f.write(f"  ROC-AUC  : {m['roc_auc']}\n")
        f.write(f"  Confusion Matrix: {m['confusion_matrix']}\n\n")
    f.write(f"Best model (by F1-score): {best_model_name}\n")
print(f"Saved {RESULTS_DIR}/metrics_summary.txt")

print("\n### PIPELINE COMPLETE ###")
