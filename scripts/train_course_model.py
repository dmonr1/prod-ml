from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.config import COURSE_MODEL_PATH
from app.schemas.training import FEATURES_CURSO


DATASET_PATH = Path("data/dataset_riesgo_curso.csv")


def construir_objetivo_fracaso(df: pd.DataFrame) -> pd.Series:
    if "fracaso_curso" in df.columns:
        return df["fracaso_curso"].fillna(0).astype(int)

    nota_curso = df.get("nota_curso", 20).fillna(20)
    return (nota_curso < 11).astype(int)


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    for column in FEATURES_CURSO:
        if column not in df.columns:
            df[column] = 0

    X = df[FEATURES_CURSO]
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

    print("=== REPORTE DE CLASIFICACION CURSO ===")
    print(classification_report(y_test, y_pred, target_names=["NO_FRACASO", "FRACASO"]))

    COURSE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, COURSE_MODEL_PATH)

    print(f"Modelo curso guardado en: {COURSE_MODEL_PATH}")


if __name__ == "__main__":
    main()
