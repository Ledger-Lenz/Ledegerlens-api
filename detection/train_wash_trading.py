"""
Train and evaluate RF / XGBoost / LightGBM classifiers with SMOTE
against the labeled wash-trading dataset.

Sub-issue #13 — sub-issue 2/3 of the #8 ML ensemble epic.
Repo: Ledger-Lenz/Ledegerlens-api
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    SMOTE = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

# Optional imports — gated behind availability checks
HAS_XGB = False
HAS_LGB = False


def _import_xgb():
    """Lazy-import xgboost; returns module or None."""
    try:
        import xgboost as _xgb
        return _xgb
    except Exception:
        return None


def _import_lgb():
    """Lazy-import lightgbm; returns module or None."""
    try:
        import lightgbm as _lgb
        return _lgb
    except Exception:
        return None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("train_wash_trading")


# ── Config ───────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_path: Path = Path("data/labeled_wash_trading.csv")
    model_dir: Path = Path("models")
    random_state: int = 42
    n_folds: int = 5
    smote_k_neighbors: int = 5
    smote_sampling_strategy: float | str = "auto"
    test_size: float = 0.2
    # RF
    rf_n_estimators: int = 200
    rf_max_depth: int | None = 15
    # XGBoost
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    # LightGBM
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 8
    lgb_learning_rate: float = 0.05


# ── Feature extraction (mirrors detection/feature_engineering.py) ────────

def extract_features(df: pd.DataFrame) -> np.ndarray:
    """Extract a flat feature matrix from the labeled dataset.

    If the dataset already contains pre-extracted features, this is a
    pass-through.  Otherwise it computes the standard LedgerLens features:
    counterparty concentration, round-trip frequency, volume ratios, etc.
    """
    feature_cols = [c for c in df.columns if c.startswith("feat_") or c.startswith("f_")]
    if feature_cols:
        return df[feature_cols].values.astype(np.float32)

    # Fallback: extract from raw trade columns.
    LOGGER.info("No pre-extracted features found — computing from raw columns.")
    feats = []
    # Structural features (present in most labeled datasets)
    for col in [
        "volume_usd", "trade_count", "unique_counterparties",
        "counterparty_concentration", "round_trip_freq",
        "self_matching_rate", "volume_per_counterparty",
        "intra_minute_trade_ratio", "off_hours_ratio",
        "avg_trade_interval_seconds", "std_trade_interval_seconds",
        "benford_score", "age_days", "funding_source_count",
    ]:
        if col in df.columns:
            feats.append(df[col].fillna(0).values.astype(np.float32))
    if not feats:
        raise ValueError("No feature columns found in dataset.")
    return np.column_stack(feats)


def load_data(config: TrainConfig) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load labeled data, return (X, y, feature_names)."""
    path = Path(config.data_path) if isinstance(config.data_path, str) else config.data_path
    if not path.exists():
        raise FileNotFoundError(f"Labeled dataset not found: {path}")

    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if "label" not in df.columns:
        raise KeyError("Dataset missing 'label' column.")

    X = extract_features(df)
    y = df["label"].values.astype(int)
    feature_names = [c for c in df.columns
                     if c.startswith("feat_") or c.startswith("f_")]
    if not feature_names:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    LOGGER.info("Loaded %d samples, %d features. Label dist: %s",
                len(y), X.shape[1], dict(zip(*np.unique(y, return_counts=True))))
    return X, y, feature_names


# ── Training & evaluation ────────────────────────────────────────────────

@dataclass
class ModelResult:
    name: str
    metrics: dict = field(default_factory=dict)
    model: object | None = None
    train_time_s: float = 0.0


def evaluate_model(name, model, X_train, X_test, y_train, y_test) -> ModelResult:
    """Fit and evaluate a single model."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_proba = (model.predict_proba(X_test)[:, 1]
               if hasattr(model, "predict_proba") else None)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        except ValueError:
            metrics["roc_auc"] = float("nan")

    LOGGER.info("%s — f1: %.4f  roc_auc: %s  time: %.2fs",
                name, metrics["f1"],
                f'{metrics.get("roc_auc", "N/A"):.4f}',
                train_time)

    return ModelResult(name=name, metrics=metrics, model=model, train_time_s=train_time)


def cross_validate_model(name, model, X, y, cv) -> dict:
    """Stratified k-fold cross-validation."""
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    try:
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                n_jobs=-1, error_score="raise")
    except Exception as e:
        LOGGER.warning("%s CV failed: %s", name, e)
        return {}
    return {k: float(np.mean(scores[f"test_{k}"])) for k in
            ["accuracy", "precision", "recall", "f1", "roc_auc"]}


def train_all(config: TrainConfig) -> dict[str, ModelResult]:
    """Train RF + XGBoost + LightGBM with SMOTE, return results."""
    X, y, feature_names = load_data(config)

    # Train / test split (stratified, leakage-free)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y,
    )

    # SMOTE — applied ONLY to training fold (no leakage)
    if HAS_SMOTE:
        smote = SMOTE(
            sampling_strategy=config.smote_sampling_strategy,
            k_neighbors=min(config.smote_k_neighbors, int(np.min(np.bincount(y_train))) - 1),
            random_state=config.random_state,
        )
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
        LOGGER.info("SMOTE: %d → %d samples", len(y_train), len(y_train_bal))
    else:
        X_train_bal, y_train_bal = X_train, y_train
        LOGGER.info("SMOTE not available — using raw class distribution")

    # Scale
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_test = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    results = {}

    # ── Random Forest ──
    rf = RandomForestClassifier(
        n_estimators=config.rf_n_estimators,
        max_depth=config.rf_max_depth,
        random_state=config.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    results["RandomForest"] = evaluate_model("RandomForest", rf, X_train_bal, X_test, y_train_bal, y_test)
    results["RandomForest"].metrics["cv"] = cross_validate_model("RandomForest", rf, X, y, cv)

    # ── XGBoost ──
    xgb = _import_xgb()
    if xgb is not None:
        xgb_clf = xgb.XGBClassifier(
            n_estimators=config.xgb_n_estimators,
            max_depth=config.xgb_max_depth,
            learning_rate=config.xgb_learning_rate,
            random_state=config.random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
        results["XGBoost"] = evaluate_model("XGBoost", xgb_clf, X_train_bal, X_test, y_train_bal, y_test)
        results["XGBoost"].metrics["cv"] = cross_validate_model("XGBoost", xgb_clf, X, y, cv)
    else:
        LOGGER.warning("XGBoost not available — skipping.")

    # ── LightGBM ──
    lgb = _import_lgb()
    if lgb is not None:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=config.lgb_n_estimators,
            max_depth=config.lgb_max_depth,
            learning_rate=config.lgb_learning_rate,
            random_state=config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        results["LightGBM"] = evaluate_model("LightGBM", lgb_clf, X_train_bal, X_test, y_train_bal, y_test)
        results["LightGBM"].metrics["cv"] = cross_validate_model("LightGBM", lgb_clf, X, y, cv)
    else:
        LOGGER.warning("LightGBM not available — skipping.")

    return results


# ── Save artifacts ───────────────────────────────────────────────────────

def save_results(results: dict[str, ModelResult], config: TrainConfig):
    """Persist trained models and metrics."""
    config.model_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name, r in results.items():
        if r.model is not None:
            import joblib
            path = config.model_dir / f"{name.lower()}_smote.joblib"
            joblib.dump(r.model, path)
            LOGGER.info("Saved %s → %s", name, path)
        summary[name] = {
            "metrics": {k: round(v, 6) for k, v in r.metrics.items()
                        if k != "cv" and isinstance(v, (int, float))},
            "cv": r.metrics.get("cv", {}),
            "train_time_s": round(r.train_time_s, 3),
        }

    metrics_path = config.model_dir / "evaluation_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    LOGGER.info("Evaluation summary → %s", metrics_path)

    # Print summary
    print("\n" + "=" * 60)
    print("  Wash-Trading Classifier — Evaluation Summary")
    print("=" * 60)
    for name, r in results.items():
        m = r.metrics
        print(f"\n  {name}:")
        print(f"    F1:       {m.get('f1', 'N/A'):.4f}")
        print(f"    ROC-AUC:  {m.get('roc_auc', 'N/A'):.4f}")
        print(f"    Precision:{m.get('precision', 'N/A'):.4f}")
        print(f"    Recall:   {m.get('recall', 'N/A'):.4f}")
        cv = m.get("cv", {})
        if cv:
            print(f"    CV-F1:    {cv.get('f1', 'N/A'):.4f}")
        print(f"    Train:    {r.train_time_s:.2f}s")
    print("=" * 60)


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train wash-trading classifiers")
    parser.add_argument("--data", type=Path, default=Path("data/labeled_wash_trading.csv"),
                        help="Path to labeled dataset")
    parser.add_argument("--model-dir", type=Path, default=Path("models"),
                        help="Directory to save trained models")
    parser.add_argument("--no-xgb", action="store_true", help="Skip XGBoost")
    parser.add_argument("--no-lgb", action="store_true", help="Skip LightGBM")
    args = parser.parse_args()

    if args.no_xgb:
        global HAS_XGB
        HAS_XGB = False
    if args.no_lgb:
        global HAS_LGB
        HAS_LGB = False

    config = TrainConfig(data_path=args.data, model_dir=args.model_dir)
    results = train_all(config)
    save_results(results, config)
    return results


if __name__ == "__main__":
    main()
