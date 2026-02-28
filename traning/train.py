import sys
import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt



# adding project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# load data 
df = pd.read_csv("data/features.csv")
print(f"Loaded {len(df)} rows from features.csv")

# separate features from label
X = df.drop(columns=['optimal_model'])
y_raw = df['optimal_model']

# verify feature columns are numeric 
print("Feature columns:")
print(X.dtypes)
print()

# encode labels 
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Label mapping: {dict(zip(le.classes_, range(len(le.classes_))))}")

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Test set distribution : {np.bincount(y_test)}\n")

# train XGBoost model
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train, y_train)
print("Model training complete.\n")

# evaluation 
y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {f1:.4f}")

# cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Per-fold: {[round(s, 3) for s in cv_scores]}")

# cost matrix and weighted cost score 

COST_MATRIX = np.array([
    # pred: flash gpt4o o3-mini
    [0.0, 1.0, 3.0], # true - gemini
    [5.0, 0.0, 1.5], # true - gpt-4o
    [10.0, 3.0, 0.0], # true - o3-mini
])

def weighted_cost_score(y_true, y_pred):
    """Average cost per prediction using the cost matrix."""
    total = sum(COST_MATRIX[t][p] for t,p in zip(y_true, y_pred))
    return round(total / len(y_true), 4)

cost  = weighted_cost_score(y_test, y_pred)
print(f"Weighted cost score: {cost} (the lower the better)\n")

# feature importance chart 
os.makedirs("traning", exist_ok=True)

importances = model.feature_importances_
feature_names = X.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(feature_names[sorted_idx], importances[sorted_idx])
plt.xlabel("Importance")
plt.title("XGBoost Feature Importance - LLM Router")
plt.tight_layout()
plt.savefig("traning/feature_importance.png", dpi=150)
print("Saved traning/feature_importance.png")

# save model and label encoder 
os.makedirs("models", exist_ok=True)

# Use joblib to save the full sklearn wrapper (save_model has a bug in xgboost 2.1+)
joblib.dump(model, "models/router_v1.joblib")
print("Saved models/router_v1.joblib")

with open("models/label_classes.json", "w") as f:
    json.dump(le.classes_.tolist(), f)
print("Saved models/label_classes.json")

# save the feature column order (for the API)
with open("models/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)
print("Saved models/feature_columns.json")

print("All done!")