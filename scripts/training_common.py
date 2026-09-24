from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold


METADATA_COLUMNS = {"alumno_id", "fecha_corte", "fecha_resultado"}


def entrenar_modelo_temporal(dataset_path: Path, target: str, features: list[str], model_path: Path) -> None:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_path}")

    data = pd.read_csv(dataset_path)
    required = METADATA_COLUMNS | {target} | set(features)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            "El dataset no es apto para validar alerta temprana. Faltan columnas: "
            + ", ".join(missing)
            + ". Cada fila debe contener variables observadas al corte, el alumno y el resultado final posterior."
        )

    data["fecha_corte"] = pd.to_datetime(data["fecha_corte"], errors="coerce")
    data["fecha_resultado"] = pd.to_datetime(data["fecha_resultado"], errors="coerce")
    if data[["fecha_corte", "fecha_resultado", "alumno_id", target, *features]].isna().any().any():
        raise ValueError("El dataset tiene fechas, identificadores, objetivos o variables predictoras vacíos/ inválidos.")
    if (data["fecha_corte"] >= data["fecha_resultado"]).any():
        raise ValueError("Cada fecha_corte debe ser anterior a fecha_resultado; se detectó fuga temporal.")

    labels = data[target].astype(int)
    if not set(labels.unique()).issubset({0, 1}) or labels.nunique() != 2:
        raise ValueError(f"{target} debe contener ambas clases binarias 0 y 1.")
    if data["alumno_id"].nunique() < 4:
        raise ValueError("Se requieren al menos cuatro alumnos distintos para una partición agrupada de entrenamiento/prueba.")

    x = data[features].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any():
        raise ValueError("Las variables predictoras deben ser numéricas y completas; no se imputarán con datos del futuro.")

    splitter = GroupShuffleSplit(n_splits=100, test_size=0.25, random_state=42)
    split = next(
        (
            (train_index, test_index)
            for train_index, test_index in splitter.split(x, labels, groups=data["alumno_id"])
            if labels.iloc[train_index].nunique() == 2 and labels.iloc[test_index].nunique() == 2
        ),
        None,
    )
    if split is None:
        raise ValueError("No se pudo formar una prueba agrupada por alumno con ambas clases; se necesitan más datos.")

    train_index, test_index = split
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    model = _crear_xgboost(y_train)
    model.fit(x.iloc[train_index], y_train)
    probabilities = model.predict_proba(x.iloc[test_index])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"=== PRUEBA AGRUPADA POR ALUMNO: {target} ===")
    print(f"Registros: {len(data)} | Alumnos: {data['alumno_id'].nunique()} | Train: {len(train_index)} | Test: {len(test_index)}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test, predictions, zero_division=0):.4f}")
    print(f"AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print("Confusion matrix [TN FP; FN TP]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_by_fold = {"accuracy": [], "f1": [], "auc": []}
    for fold_train, fold_test in folds.split(x, labels, groups=data["alumno_id"]):
        y_fold_train, y_fold_test = labels.iloc[fold_train], labels.iloc[fold_test]
        if y_fold_train.nunique() != 2 or y_fold_test.nunique() != 2:
            raise ValueError("La validación cruzada agrupada requiere ambas clases en cada fold; se necesitan más alumnos.")
        fold_model = _crear_xgboost(y_fold_train)
        fold_model.fit(x.iloc[fold_train], y_fold_train)
        fold_probability = fold_model.predict_proba(x.iloc[fold_test])[:, 1]
        fold_prediction = (fold_probability >= 0.5).astype(int)
        metrics_by_fold["accuracy"].append(accuracy_score(y_fold_test, fold_prediction))
        metrics_by_fold["f1"].append(f1_score(y_fold_test, fold_prediction, zero_division=0))
        metrics_by_fold["auc"].append(roc_auc_score(y_fold_test, fold_probability))
    print("5-fold CV (media ± desviación estándar):")
    for metric, values in metrics_by_fold.items():
        print(f"{metric}: {pd.Series(values).mean():.4f} ± {pd.Series(values).std(ddof=1):.4f}")

    final_model = _crear_xgboost(labels)
    final_model.fit(x, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Modelo XGBoost guardado en: {model_path}")


def _crear_xgboost(labels: pd.Series):
    from xgboost import XGBClassifier
    positivos = int((labels == 1).sum())
    negativos = int((labels == 0).sum())
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=negativos / positivos,
        random_state=42,
        n_jobs=1,
    )
