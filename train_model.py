"""
Cricket Win Prediction - Model Training Pipeline
Trains multiple ML models, evaluates them, and saves the best model + preprocessor.
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# -- Paths ---------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "cricket_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -- Features ------------------------------------------------------------------
CATEGORICAL_FEATURES = ["venue", "batting_team", "bowling_team", "toss_winner", "toss_decision"]
NUMERICAL_FEATURES   = [
    "first_innings_score", "first_innings_wickets", "target",
    "balls_bowled", "overs_completed", "balls_in_over",
    "runs_scored", "wickets_fallen", "runs_needed",
    "balls_left", "crr", "rrr",
]
TARGET = "batting_team_wins"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        print(" Dataset not found - generating ")
        sys.path.insert(0, os.path.dirname(path))
        from generate_dataset import generate_dataset
        df = generate_dataset(8000)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)
    print(f"[OK] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build a column transformer: OHE for categoricals, scaler for numericals."""
    cat_pipe = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    num_pipe = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("num", num_pipe, NUMERICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'-'*50}")
    print(f"  Model : {name}")
    print(f"  Accuracy : {acc:.4f}   AUC-ROC : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    return {"name": name, "accuracy": acc, "auc": auc, "confusion_matrix": cm,
            "report": report, "model": model}


def train(data_path: str = DATA_PATH):
    # -- 1. Load ---------------------------------------------------------------
    df = load_data(data_path)
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # -- 2. Preprocessor -------------------------------------------------------
    preprocessor = build_preprocessor()

    # -- 3. Models -------------------------------------------------------------
    models_config = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=4,
            random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
    }

    # -- 4. Train & Evaluate ---------------------------------------------------
    results = []
    for name, clf in models_config.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier",   clf),
        ])
        pipeline.fit(X_train, y_train)
        res = evaluate_model(name, pipeline, X_test, y_test)
        results.append(res)

    # -- 5. Pick Best ----------------------------------------------------------
    best = max(results, key=lambda r: r["auc"])
    print(f"\n[BEST] Best Model: {best['name']} (AUC={best['auc']:.4f})")

    # -- 6. Feature Importance -------------------------------------------------
    feature_names = (
        best["model"].named_steps["preprocessor"]
        .get_feature_names_out()
        .tolist()
    )
    clf_step = best["model"].named_steps["classifier"]
    if hasattr(clf_step, "feature_importances_"):
        importances = clf_step.feature_importances_
        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(20)
    else:
        fi_df = pd.DataFrame(columns=["feature", "importance"])

    # -- 7. Save Artifacts -----------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, "model_metadata.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)

    metadata = {
        "model_name":         best["name"],
        "accuracy":           best["accuracy"],
        "auc":                best["auc"],
        "confusion_matrix":   best["confusion_matrix"],
        "report":             best["report"],
        "feature_importance": fi_df,
        "all_results":        [
            {"name": r["name"], "accuracy": r["accuracy"], "auc": r["auc"]}
            for r in results
        ],
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features":   NUMERICAL_FEATURES,
        "target":               TARGET,
    }
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n[SAVE] Model saved   -> {model_path}")
    print(f"[SAVE] Metadata saved -> {meta_path}")
    return best["model"], metadata


if __name__ == "__main__":
    train()
