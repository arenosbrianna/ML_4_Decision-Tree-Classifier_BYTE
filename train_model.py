"""Trains the decision tree pipeline and saves it for the Streamlit app.

Mirrors the preprocessing/training steps in bank_marketing_decision_tree.ipynb.
Run this once (or whenever the dataset/model changes): `python train_model.py`.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bank-additional-full.csv")
MODEL_PATH = os.path.join("model", "decision_tree_pipeline.joblib")
METRICS_PATH = os.path.join("model", "metrics.json")

CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUMERIC_COLS = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]


def main():
    df = pd.read_csv(DATA_PATH, sep=";").drop(columns=["duration"])
    X = df[CATEGORICAL_COLS + NUMERIC_COLS]

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERIC_COLS),
        ]
    )
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    top5 = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)[:5]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=target_encoder.classes_
        ),
        "top5_features": [{"feature": f, "importance": float(i)} for f, i in top5],
        "target_classes": target_encoder.classes_.tolist(),
    }

    os.makedirs("model", exist_ok=True)
    joblib.dump({"pipeline": model, "target_encoder": target_encoder}, MODEL_PATH)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
