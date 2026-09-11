"""
Training pipeline for FloodLens.

Steps:
  1. Load and clean data (real if present, else synthetic fallback)
  2. Feature engineering & target construction
  3. Chronologically-agnostic but stratified train/val/test split
     (data has no time column, so we use stratified splits; documented)
  4. Preprocessing (standardize numerics, one-hot categoricals)
  5. Train & compare Logistic Regression, Decision Tree, Random Forest,
     Gradient Boosting, XGBoost (optional)
  6. Tune the best model with RandomizedSearchCV
  7. Save model, preprocessing pipeline, metadata, metrics, feature list
     to the models/ directory using joblib + JSON
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import Config  # noqa: E402
from src.data_processing import load_dataset, clean_data  # noqa: E402
from src.feature_engineering import (  # noqa: E402
    engineer_pipeline, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    BINARY_FEATURES, RISK_LABELS,
)

logger = logging.getLogger("floodlens.train")

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """Preprocessing pipeline that will be embedded inside the final model
    pipeline. This prevents data leakage because it is fit only on the
    training fold."""
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    binary_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe, CATEGORICAL_FEATURES),
            ("bin", binary_pipe, BINARY_FEATURES),
        ],
        remainder="drop",
    )
    return pre


def get_models() -> Dict[str, Any]:
    """Return a dict of candidate classifier instances."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, random_state=RANDOM_STATE,
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.info("XGBoost not available (%s); skipping.", exc)
    return models


def evaluate_model(model, X, y) -> Dict[str, float]:
    """Compute key metrics."""
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision_macro": float(precision_score(y, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, preds, average="weighted", zero_division=0)),
    }


def tune_model(name: str, base_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series):
    """Lightweight random search."""
    pre_params = {}  # keep preprocessor fixed
    grid: Dict[str, Any] = {}
    if name == "Random Forest":
        grid = {
            "classifier__n_estimators": [200, 300],
            "classifier__max_depth": [12, 20, None],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", "log2"],
        }
    elif name == "Gradient Boosting":
        grid = {
            "classifier__n_estimators": [150, 250],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.05, 0.1],
        }
    elif name == "XGBoost":
        grid = {
            "classifier__n_estimators": [150, 250],
            "classifier__max_depth": [4, 6],
            "classifier__learning_rate": [0.05, 0.1],
        }
    elif name == "Logistic Regression":
        grid = {
            "classifier__C": [0.1, 1.0, 10.0],
        }
    elif name == "Decision Tree":
        grid = {
            "classifier__max_depth": [8, 12, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
        }
    else:
        return base_pipeline, {}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_pipeline, grid, n_iter=min(10, max(1, len(grid)) * 2),
        scoring="f1_macro", cv=cv, n_jobs=-1, random_state=RANDOM_STATE,
        refit=True,
    )
    logger.info("Tuning %s ...", name)
    search.fit(X_train, y_train)
    logger.info("%s best params: %s (CV f1_macro=%.4f)",
                name, search.best_params_, search.best_score_)
    return search.best_estimator_, {
        "best_params": {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
                        for k, v in search.best_params_.items()},
        "best_cv_f1_macro": float(search.best_score_),
    }


def _extract_feature_names(model_pipe: Pipeline) -> list:
    """Attempt to extract final feature names after one-hot encoding."""
    try:
        pre: ColumnTransformer = model_pipe.named_steps["preprocessor"]
        names: list = []
        for name, trans, cols in pre.transformers_:
            if name == "cat":
                ohe = trans.named_steps["onehot"]
                names.extend([f"{c}={v}" for c, vs in zip(cols, ohe.categories_) for v in vs])
            else:
                names.extend(list(cols))
        return names
    except Exception:
        return []


def run_training(force_synthetic: bool = False) -> Dict[str, Any]:
    """Full training pipeline. Returns a metadata dict."""
    Config.ensure_dirs()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    t0 = time.time()

    # 1. Load and clean
    df_raw, is_synthetic = load_dataset(force_synthetic=force_synthetic)
    df_clean = clean_data(df_raw)

    # 2. Feature engineering
    df, feature_names = engineer_pipeline(df_clean)

    # 3. Train/test split (stratified)
    X = df[feature_names].copy()
    y = df["Risk_Level_Int"].copy()

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.1765,  # ~15% of total for val
        random_state=RANDOM_STATE, stratify=y_tv,
    )
    logger.info("Split sizes: train=%d val=%d test=%d", len(X_train), len(X_val), len(X_test))

    # 4-5. Train and compare models (without heavy tuning first)
    models = get_models()
    results = {}
    for name, clf in models.items():
        logger.info("Training %s ...", name)
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        val_metrics = evaluate_model(pipe, X_val, y_val)
        results[name] = {"val": val_metrics, "pipeline": pipe}
        logger.info("%s -> val f1_macro=%.4f", name, val_metrics["f1_macro"])

    # 6. Pick best based on val f1_macro (primary) with recall tie-break
    best_name = max(
        results.keys(),
        key=lambda k: (results[k]["val"]["f1_macro"], results[k]["val"]["recall_macro"]),
    )
    logger.info("Best candidate before tuning: %s", best_name)

    best_pipe = results[best_name]["pipeline"]
    tuned_pipe, tune_info = tune_model(best_name, best_pipe, X_train, y_train)

    # Final evaluation on test set
    test_metrics = evaluate_model(tuned_pipe, X_test, y_test)
    preds_test = tuned_pipe.predict(X_test)
    proba_test = tuned_pipe.predict_proba(X_test)
    cm = confusion_matrix(y_test, preds_test, labels=list(range(len(RISK_LABELS))))
    cls_report = classification_report(
        y_test, preds_test, target_names=RISK_LABELS, output_dict=True, zero_division=0,
    )
    logger.info("Test metrics for tuned %s: %s", best_name, test_metrics)

    # Save artifacts
    model_path = Config.MODELS_DIR / Config.MODEL_FILE
    prep_path = Config.MODELS_DIR / Config.PREPROCESSOR_FILE
    joblib.dump(tuned_pipe, model_path)
    # Also save preprocessor standalone for potential reuse
    joblib.dump(tuned_pipe.named_steps["preprocessor"], prep_path)

    # Feature importances (if available)
    final_feature_names = _extract_feature_names(tuned_pipe)
    classifier = tuned_pipe.named_steps["classifier"]
    importances = None
    if hasattr(classifier, "feature_importances_"):
        importances = [float(v) for v in classifier.feature_importances_]
    elif hasattr(classifier, "coef_"):
        importances = [float(np.linalg.norm(classifier.coef_[:, i]))
                       for i in range(classifier.coef_.shape[1])]

    metadata = {
        "model_version": Config.MODEL_VERSION,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - t0, 2),
        "is_synthetic_data": bool(is_synthetic),
        "dataset": {
            "source": "Kaggle 'Flood Risk in India' (CC0)" if not is_synthetic
                      else "Synthetic (pipeline-generated, for demo only)",
            "url": "https://www.kaggle.com/datasets/s3programmer/flood-risk-in-india" if not is_synthetic
                   else None,
            "license": "CC0 Public Domain" if not is_synthetic else "N/A (synthetic)",
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "class_distribution": {k: int(v) for k, v in df["Risk_Level"].value_counts().items()},
            "feature_names": list(feature_names),
            "numeric_features": list(NUMERIC_FEATURES),
            "categorical_features": list(CATEGORICAL_FEATURES),
            "binary_features": list(BINARY_FEATURES),
            "risk_labels": list(RISK_LABELS),
        },
        "model_selection": {
            "candidates": {k: v["val"] for k, v in results.items()},
            "selected_model": best_name,
            "tuning": tune_info,
        },
        "test_metrics": test_metrics,
        "classification_report": cls_report,
        "confusion_matrix": cm.tolist(),
        "feature_names_transformed": final_feature_names,
        "feature_importances": importances,
        "artifacts": {
            "model": str(model_path),
            "preprocessor": str(prep_path),
        },
    }

    meta_path = Config.MODELS_DIR / Config.METADATA_FILE
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=float)
    logger.info("Saved metadata to %s", meta_path)

    # Also persist engineered training data for dashboard use
    df.to_csv(Config.PROCESSED_DATA_DIR / "engineered.csv", index=False)
    X_test.assign(y_true=y_test.values, y_pred=preds_test).to_csv(
        Config.PROCESSED_DATA_DIR / "test_predictions.csv", index=False,
    )
    return metadata


if __name__ == "__main__":
    force_synth = "--synthetic" in sys.argv
    meta = run_training(force_synthetic=force_synth)
    print(json.dumps(meta["test_metrics"], indent=2))
