import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import kagglehub
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

print("--- Step 1: Loading Data ---")
path = kagglehub.dataset_download("ritwikb3/heart-disease-cleveland")
print("Path to dataset files:", path)

csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No CSV file found in the downloaded dataset.")

file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\n--- Step 2: Exploratory Data Analysis ---")
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

target_col = 'target' if 'target' in df.columns else 'condition'

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target_col, palette='Set2')
plt.title("Heart Disease Distribution (0 = No Disease, 1 = Disease)")
plt.savefig('plot_target_distribution.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.savefig('plot_correlation_heatmap.png', bbox_inches='tight')
plt.close()

print("\n--- Step 3: Preprocessing ---")

print(f"\nMissing values before handling:\n{df.isnull().sum()}")
df = df.fillna(df.mean())
print("\nMissing values after handling:", df.isnull().sum().sum())

df['ca_thal_risk'] = df['ca'] * df['thal']
df['ca_high']      = (df['ca'] >= 2).astype(int)
df['oldpeak_ca']   = df['oldpeak'] * df['ca']
df['thalach_age']  = df['thalach'] / df['age']

categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
cat_cols = [col for col in categorical_features if col in df.columns]
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Step 4: Model Training & Evaluation ---")

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector Machine": SVC(probability=True, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

best_model_name = ""
best_model = None
best_accuracy = 0

for name, model in models.items():
    print(f"\nEvaluating: {name}")
    print("-" * 30)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if roc_auc > best_accuracy:
        best_accuracy = roc_auc
        best_model_name = name
        best_model = model

plt.figure(figsize=(10, 8))
for name, model in models.items():
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{name}")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves Comparison")
plt.legend()
plt.savefig('plot_roc_curves.png', bbox_inches='tight')
plt.close()

print("\n--- Step 6: Saving the Best Model ---")
print(f"The best model is {best_model_name} with an accuracy of {best_accuracy:.4f}")

model_filename = "best_heart_disease_model.pkl"
scaler_filename = "standard_scaler.pkl"
columns_filename = "model_columns.pkl"

joblib.dump(best_model, model_filename)
joblib.dump(scaler, scaler_filename)
joblib.dump(list(X.columns), columns_filename)

print(f"Model saved to: {model_filename}")
print(f"Scaler saved to: {scaler_filename}")
print(f"Columns saved to: {columns_filename}")
print("\nProject Complete! You can load the model later using joblib.load('filename')")
