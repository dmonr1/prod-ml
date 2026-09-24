from pathlib import Path

from app.config import GLOBAL_MODEL_PATH
from app.schemas.training import FEATURES_GLOBALES_CORTE
from scripts.training_common import entrenar_modelo_temporal


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "dataset_riesgo_global.csv"


def main() -> None:
    entrenar_modelo_temporal(DATASET_PATH, "fracaso_global", FEATURES_GLOBALES_CORTE, GLOBAL_MODEL_PATH)


if __name__ == "__main__":
    main()
