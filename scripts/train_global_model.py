from pathlib import Path

import joblib
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

from app.config import GLOBAL_MODEL_PATH, PCA_PATH
from app.schemas.training import FEATURES_GLOBALES, TARGET_GLOBAL


DATASET_PATH = Path("data/dataset_riesgo_global.csv")
ENCODER_PATH = Path("trained_models/label_encoder_global.joblib")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    for column in FEATURES_GLOBALES:
        if column not in df.columns:
            df[column] = 0

    X = df[FEATURES_GLOBALES]
    y = df[TARGET_GLOBAL]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(5, len(FEATURES_GLOBALES)))),
            (
                "model",
                XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob",
                    eval_metric="mlogloss",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("=== REPORTE DE CLASIFICACION ===")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    GLOBAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, GLOBAL_MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    pca_step = pipeline.named_steps["pca"]
    joblib.dump(pca_step, PCA_PATH)

    print(f"Modelo global guardado en: {GLOBAL_MODEL_PATH}")
    print(f"PCA guardado en: {PCA_PATH}")
    print(f"Encoder guardado en: {ENCODER_PATH}")


if __name__ == "__main__":
    main()
