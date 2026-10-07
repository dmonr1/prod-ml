"""Measured evaluation on a student-disjoint holdout; writes staged artifacts only."""
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import MODEL_TRAINING_SOURCE
from app.services.artifact_service import write_json

SEED = 42
METADATA_COLUMNS = {"alumno_id", "fecha_corte", "fecha_resultado"}


def load_dataset(dataset_path: Path, target: str, features: list[str]):
    source_bytes = dataset_path.read_bytes()
    data = pd.read_csv(BytesIO(source_bytes))
    data.attrs["dataset_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    required = METADATA_COLUMNS | {target} | set(features)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError("Dataset incompatible con el modelo de cortes. Faltan: " + ", ".join(missing))
    if data[list(required)].isna().any().any():
        raise ValueError("El dataset contiene datos requeridos vacíos.")
    # Older exports may include the marker; current CSV files keep provenance in run metadata.
    if "origen_datos" in data and set(data["origen_datos"].unique()) != {"SIMULADO_SOLO_PARA_PRUEBAS"}:
        raise ValueError("El origen declarado no coincide con el entrenamiento sintético configurado.")
    for column in ("fecha_corte", "fecha_resultado"):
        data[column] = pd.to_datetime(data[column], errors="raise")
    if (data["fecha_corte"] >= data["fecha_resultado"]).any():
        raise ValueError("Cada fecha_corte debe ser anterior a fecha_resultado.")
    labels = pd.to_numeric(data[target], errors="raise")
    if set(labels.unique()) != {0, 1}:
        raise ValueError(f"{target} debe contener ambas clases binarias 0 y 1, sin valores fraccionarios.")
    groups = data["alumno_id"].astype(str)
    if groups.str.strip().eq("").any() or groups.nunique() < 8:
        raise ValueError("Se requieren identificadores válidos y al menos ocho alumnos distintos.")
    x = data[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Los predictores deben contener números finitos.")
    row_key = ["alumno_id", "fecha_corte"]
    outcome_key = ["alumno_id", "fecha_resultado"]
    if target == "fracaso_curso":
        if "curso_id" not in data or data["curso_id"].isna().any():
            raise ValueError("El dataset de cursos requiere curso_id.")
        row_key.append("curso_id")
        outcome_key.append("curso_id")
    if data.duplicated(row_key).any():
        raise ValueError("Hay filas duplicadas para el mismo alumno, curso y corte.")
    if (data.groupby(outcome_key)[target].nunique() > 1).any():
        raise ValueError("El resultado final debe ser consistente entre cortes de un mismo alumno y curso.")
    return data, x, labels.astype(int), groups


def _crear_xgboost(labels: pd.Series):
    return XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", n_estimators=120,
        max_depth=3, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=float((labels == 0).sum() / (labels == 1).sum()),
        random_state=SEED, n_jobs=1,
    )


def _models(labels):
    return {
        "xgboost": ("XGBoost", "Gradient boosted trees", _crear_xgboost(labels)),
        "random_forest": ("Random Forest", "Bagging", RandomForestClassifier(
            n_estimators=150, max_depth=6, class_weight="balanced", random_state=SEED, n_jobs=1)),
        "gradient_boosting": ("Gradient Boosting", "Gradient boosted trees", GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)),
        "logistic_regression": ("Regresión logística", "Modelo lineal", make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED))),
    }


def metrics(labels, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1_score": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def entrenar_modelo_temporal(dataset_path: Path, target: str, features: list[str], model_path: Path) -> dict:
    data, x, labels, groups = load_dataset(dataset_path, target, features)
    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=SEED)
    split = next(((tr, te) for tr, te in splitter.split(x, labels, groups)
                  if labels.iloc[tr].nunique() == labels.iloc[te].nunique() == 2), None)
    if split is None:
        raise ValueError("No fue posible separar alumnos con ambas clases en entrenamiento y prueba.")
    train, test = split
    train_groups, test_groups = groups.iloc[train], groups.iloc[test]
    assert not set(train_groups) & set(test_groups)
    x_train, y_train = x.iloc[train], labels.iloc[train]
    x_test, y_test = x.iloc[test], labels.iloc[test]
    if train_groups.nunique() < 5:
        raise ValueError("La validación cruzada requiere cinco alumnos de entrenamiento.")
    # Holdout students never enter CV. Scalers in baselines are fitted on training only.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = []
    for number, (tr, va) in enumerate(cv.split(x_train, y_train, train_groups), 1):
        if y_train.iloc[tr].nunique() != 2 or y_train.iloc[va].nunique() != 2:
            raise ValueError("Cada fold requiere ambas clases; se necesitan más alumnos.")
        model = _crear_xgboost(y_train.iloc[tr])
        model.fit(x_train.iloc[tr], y_train.iloc[tr])
        folds.append({"fold": number, "train_students": sorted(set(train_groups.iloc[tr])),
                      "validation_students": sorted(set(train_groups.iloc[va])),
                      **metrics(y_train.iloc[va], model.predict_proba(x_train.iloc[va])[:, 1])})
    algorithms = []
    predictions = data.iloc[test][["alumno_id", "fecha_corte", "fecha_resultado", target]].copy()
    if "curso_id" in data:
        predictions["curso_id"] = data.iloc[test]["curso_id"]
    importance = {}
    for key, (name, family, model) in _models(y_train).items():
        model.fit(x_train, y_train)
        probability = model.predict_proba(x_test)[:, 1]
        started = perf_counter()
        for _ in range(3):
            model.predict_proba(x_test)
        milliseconds = (perf_counter() - started) * 1000 / (3 * len(test))
        estimator = model.steps[-1][1] if hasattr(model, "steps") else model
        params = estimator.get_params()
        kept = {k: params[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "scale_pos_weight", "class_weight", "C", "max_iter", "random_state") if k in params}
        if hasattr(model, "steps"):
            kept["preprocessing"] = "StandardScaler fitted on training only"
        result = metrics(y_test, probability)
        algorithms.append({"id": key, "nombre": name, "familia": family, **result,
                           "latencia_ms": milliseconds, "hiperparametros": kept,
                           "estado": "ACTIVO" if key == "xgboost" else "BASELINE" if key == "logistic_regression" else "CANDIDATO"})
        predictions[f"{key}_probability"] = probability
        if key == "xgboost":
            importance = dict(zip(features, map(float, model.feature_importances_)))
        print(f"{target} / {key}: F1={result['f1_score']:.4f}, AUC={result['roc_auc']:.4f}", flush=True)
    algorithms.sort(key=lambda item: item["f1_score"], reverse=True)
    for rank, item in enumerate(algorithms, 1):
        item["ranking"] = rank
    final_model = _crear_xgboost(labels)
    final_model.fit(x, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    predictions.to_csv(model_path.with_suffix(".predictions.csv"), index=False)
    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "source": MODEL_TRAINING_SOURCE,
        "dataset": dataset_path.name, "dataset_sha256": data.attrs["dataset_sha256"],
        "target": target, "features": features, "rows": len(data), "students": int(groups.nunique()),
        "class_counts": {str(k): int(v) for k, v in labels.value_counts().items()},
        "train_rows": len(train), "test_rows": len(test),
        "train_students": sorted(set(train_groups)), "test_students": sorted(set(test_groups)),
        "split": "GroupShuffleSplit(test_size=0.25, random_state=42); disjoint students",
        "threshold": 0.5, "algorithms": algorithms,
        "cv": {"scope": "training partition only", "folds": folds, "std_ddof": 1,
               "summary": {m: {"mean": float(np.mean([f[m] for f in folds])),
                               "std": float(np.std([f[m] for f in folds], ddof=1))}
                           for m in ("accuracy", "f1_score", "roc_auc")}},
        "feature_importance_gain": importance,
        "deployed_parameters": {k: v for k, v in final_model.get_params().items()
                                if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "scale_pos_weight", "random_state")},
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "metric_scope": "Raw classifiers before API score adjustments; not institutional validation.",
        "latency_scope": "Batch prediction time / test rows, mean of three calls, ms per row.",
    }
    write_json(model_path.with_suffix(".report.json"), report)
    return report
