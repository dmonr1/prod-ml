"""Train the API models using the generated synthetic cutoff datasets."""

from pathlib import Path

from app.config import COURSE_MODEL_PATH, GLOBAL_MODEL_PATH
from app.schemas.training import FEATURES_CURSO_CORTE, FEATURES_GLOBALES_CORTE
from scripts.training_common import entrenar_modelo_temporal


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "synthetic_demo"
def main() -> None:
    global_demo = DATA_DIR / "dataset_riesgo_global_sintetico.csv"
    course_demo = DATA_DIR / "dataset_riesgo_curso_sintetico.csv"
    if not global_demo.exists() or not course_demo.exists():
        raise FileNotFoundError("Primero ejecuta python -m scripts.generate_synthetic_cutoff_datasets")

    entrenar_modelo_temporal(
        global_demo,
        "fracaso_global",
        FEATURES_GLOBALES_CORTE,
        GLOBAL_MODEL_PATH,
    )
    entrenar_modelo_temporal(
        course_demo,
        "fracaso_curso",
        FEATURES_CURSO_CORTE,
        COURSE_MODEL_PATH,
    )
    print("La API ya carga estos artefactos. Fueron entrenados con datos sinteticos y no estan validados con alumnos reales.")


if __name__ == "__main__":
    main()
