from pathlib import Path

from app.config import COURSE_MODEL_PATH
from app.schemas.training import FEATURES_CURSO_CORTE
from scripts.training_common import entrenar_modelo_temporal


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "dataset_riesgo_curso.csv"


def main() -> None:
    entrenar_modelo_temporal(DATASET_PATH, "fracaso_curso", FEATURES_CURSO_CORTE, COURSE_MODEL_PATH)


if __name__ == "__main__":
    main()
