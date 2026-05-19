from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import GLOBAL_MODEL_PATH
from app.schemas.training import FEATURES_GLOBALES


DATASET_PATH = Path("data/dataset_riesgo_global.csv")


def construir_objetivo_fracaso(df: pd.DataFrame) -> pd.Series:
    if "fracaso_global" in df.columns:
        return df["fracaso_global"].fillna(0).astype(int)

    cursos_desaprobados = df.get("cantidad_cursos_desaprobados", 0).fillna(0)
    promedio_general = df.get("promedio_general", 20).fillna(20)
    nota_minima = df.get("nota_minima", 20).fillna(20)

    return (
        (cursos_desaprobados >= 1)
        | (promedio_general < 11)
        | (nota_minima < 11)
    ).astype(int)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    for column in FEATURES_GLOBALES:
        if column not in df.columns:
            df[column] = 0

    X = df[FEATURES_GLOBALES]
    y = construir_objetivo_fracaso(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== REPORTE DE CLASIFICACION ===")
    print(classification_report(y_test, y_pred, target_names=["NO_FRACASO", "FRACASO"]))

    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, GLOBAL_MODEL_PATH)

    print(f"Modelo global guardado en: {GLOBAL_MODEL_PATH}")


if __name__ == "__main__":
    main()
