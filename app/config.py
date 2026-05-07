from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

GLOBAL_MODEL_PATH = TRAINED_MODELS_DIR / "modelo_riesgo_global.joblib"
COURSE_MODEL_PATH = TRAINED_MODELS_DIR / "modelo_riesgo_curso.joblib"
PCA_PATH = TRAINED_MODELS_DIR / "pca_transformer.joblib"
COURSE_PCA_PATH = TRAINED_MODELS_DIR / "pca_transformer_curso.joblib"

RISK_THRESHOLDS = {
    "bajo_max": 39.99,
    "medio_max": 69.99,
}
