import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import joblib
import json
import logging

logging.getLogger("mlflow").setLevel(logging.ERROR)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# cost matrix 

COST_MATRIX = np.array([
    [0.0, 1.0, 3.0], # true: gemini
    [5.0, 0.0, 1.5], # true: gpt-4o
    [10.0, 3.0, 0.0], # true: o3-mini
])

def weighted_cost_score(y_true, y_pred):
    """Average cost per prediction using the cost matrix."""
    total = sum(COST_MATRIX[t][p] for t,p in zip(y_true, y_pred))
    return round(total / len(y_true), 4)

# loading data 

df = pd.read_csv("data/features.csv")
X = df.drop(columns=["optimal_model"])
y_raw = df["optimal_model"]

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y,
    )

feature_cols = list(X.columns)
print(f"Data: {len(df)} rows, {len(feature_cols)} features")
print(f"Classes {list(le.classes_)}\n")

# ML flow experiment

mlflow.set_experiment("llm-router-v1")

def run_experiment(run_name: str, params: dict):
    """Train one XGBoost model, log everything in MLflow."""

    with mlflow.start_run(run_name=run_name):
        
        # 1. log params 
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("data_rows", len(X_train))
        mlflow.log_param("n_features", len(feature_cols))

        # 2. train model
        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            eval_metric='mlogloss',
            random_state=42
        )
        model.fit(X_train, y_train)

        # 3. predict 
        y_pred = model.predict(X_test)

        # 4. metrics 
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cost = weighted_cost_score(y_test, y_pred)

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("weighted_cost", cost)

        # 5. log model
        mlflow.sklearn.log_model(model, "model")

        # 6. feature importance chart 
        os.makedirs("training", exist_ok=True)
        importances = model.feature_importances_
        sorted_idx = importances.argsort()

        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.barh(np.array(feature_cols)[sorted_idx],
                 importances[sorted_idx])
        ax1.set_xlabel("Feature Importance")
        ax1.set_title(f"XGBoost Feature Importances - {run_name}")
        fig1.tight_layout()
        path1 = f"training/feat_imp_{run_name}.png"
        fig1.savefig(path1, dpi=150)
        plt.close(fig1)
        mlflow.log_artifact(path1)

        # 7. confusion matrix
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=le.classes_, ax=ax2
        )
        fig2.tight_layout()
        path2 = f"training/conf_matrix_{run_name}.png"
        fig2.savefig(path2, dpi=150)
        plt.close(fig2)
        mlflow.log_artifact(path2)

        # 8. classification report 
        report = classification_report(y_test, y_pred,
                                       target_names=le.classes_)
        path3 = f"training/class_report_{run_name}.txt"
        with open(path3, "w") as f:
            f.write(report)
        mlflow.log_artifact(path3)

        print(f"  {run_name:25s} acc={acc:.3f}, f1={f1:.3f}, cost={cost}")

    return acc, f1, cost, model

# experiments to run 

experiments = [
{
    "name": "baseline",
    "params": {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
},
{
    "name": "deeper",
    "params": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
},
{
    "name": "shallow",
    "params": {
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
},
{
    "name": "more-trees-slow-lr",
    "params": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
},
{
    "name": "deep-more-trees",
    "params": {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },
},
]

# run all and save the best one 

if __name__ == "__main__":
    print("Running experiments...\n")
    print(f"  {'Name':25s} {'Accuracy':>6s} {'F1 Score':>6s} {'Cost':>6s}")
    print(" " + "-"*50)

    best_acc = 0
    best_model = None
    best_name = ""

    for exp in experiments:
        acc, f1, cost, trained_model = run_experiment(exp["name"], exp["params"])
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_name = exp["name"]
    
    print(f"\nBest model: {best_name} with accuracy {best_acc:.4f}")

    # save the best model for the api 
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/router_v1.joblib")
    print("Saved best model to models/router_v1.joblib")

    with open("models/label_classes.json", "w") as f:
        json.dump(le.classes_.tolist(), f)

    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_cols, f)

    print("Done! View results:")
    print("  mlflow ui --port 5000")
    print("Then open http://localhost:5000 in your browser.")