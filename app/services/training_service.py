"""Execute training and publish only fully completed model generations."""
from datetime import datetime, timezone
from uuid import uuid4

from app import config
from app.schemas.training import FEATURES_CURSO_CORTE, FEATURES_GLOBALES_CORTE
from app.services.artifact_service import active_run, training_lock, write_json
from scripts.training_common import entrenar_modelo_temporal


def train_models(tasks: tuple[str, ...] = ("global", "course")) -> dict:
    definitions = {
        "global": (config.GLOBAL_DATASET_PATH, "fracaso_global", FEATURES_GLOBALES_CORTE),
        "course": (config.COURSE_DATASET_PATH, "fracaso_curso", FEATURES_CURSO_CORTE),
    }
    if not tasks or set(tasks) - definitions.keys():
        raise ValueError("Ámbito de entrenamiento inválido.")
    with training_lock():
        now = datetime.now(timezone.utc)
        run_id = now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        status = {"status": "EN_EJECUCION", "started_at": now.isoformat(), "run_id": run_id}
        write_json(config.TRAINING_STATUS_PATH, status)
        try:
            models = dict(active_run().get("models", {}))
            run_dir = config.TRAINED_MODELS_DIR / "runs" / run_id
            for task in tasks:
                dataset, target, features = definitions[task]
                model_path = run_dir / f"{task}.joblib"
                report = entrenar_modelo_temporal(dataset, target, features, model_path)
                models[task] = {
                    "model_path": model_path.relative_to(config.TRAINED_MODELS_DIR).as_posix(),
                    "report_path": model_path.with_suffix(".report.json").relative_to(config.TRAINED_MODELS_DIR).as_posix(),
                    "rows": report["rows"], "features": features, "run_id": run_id,
                }
            manifest = {"run_id": run_id, "completed_at": datetime.now(timezone.utc).isoformat(),
                        "source": config.MODEL_TRAINING_SOURCE, "models": models}
            # One atomic pointer switches the generation; failed training keeps the previous models.
            write_json(config.ACTIVE_RUN_PATH, manifest)
            write_json(config.TRAINING_STATUS_PATH, {**status, "status": "COMPLETADO_EXITOSO",
                                                    "completed_at": manifest["completed_at"]})
            return manifest
        except Exception as exc:
            write_json(config.TRAINING_STATUS_PATH, {**status, "status": "FALLIDO", "error": str(exc)})
            raise
