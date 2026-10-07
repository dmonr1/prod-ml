"""Administrative views backed by persisted training reports."""
from app import config
from app.schemas.predictors import ComparativaModelosResponse, PlanificadorReentrenamientoResponse
from app.services.artifact_service import active_run, model_report, read_json, training_lock, TrainingInProgressError
from app.services.training_service import train_models


def obtener_comparativa_modelos(task: str = "global") -> ComparativaModelosResponse:
    report = model_report(task)
    return ComparativaModelosResponse(
        algoritmos=report["algorithms"],
        modelo_recomendado=report["algorithms"][0]["nombre"],
        metrica_optimizada="F1 observado en prueba; parámetros fijos, sin selección automática del modelo activo",
        fecha_evaluacion=report["evaluated_at"],
        total_registros_evaluados=report["test_rows"],
        tipo_modelo="GLOBAL" if task == "global" else "CURSO",
        origen_datos=report["source"], variables=report["features"],
        registros_entrenamiento=report["train_rows"], alumnos_prueba=len(report["test_students"]),
        alcance_metricas="Clasificadores sin los ajustes de puntaje de la API; datos sintéticos.",
    )


def obtener_planificador_reentrenamiento() -> PlanificadorReentrenamientoResponse:
    manifest = active_run()
    status = read_json(config.TRAINING_STATUS_PATH)
    state = status.get("status", "SIN_EVALUACION")
    try:
        with training_lock():
            if state == "EN_EJECUCION":
                state = "INTERRUMPIDO"
    except TrainingInProgressError:
        state = "EN_EJECUCION"
    return PlanificadorReentrenamientoResponse(
        cadencia="Manual", proxima_ejecucion_programada="No programada",
        ultimo_reentrenamiento=manifest.get("completed_at", "Sin entrenamiento registrado"),
        estado_ultimo_reentrenamiento=state,
        registros_entrenamiento=sum(item["rows"] for item in manifest.get("models", {}).values()),
        modelo_actual_version=manifest.get("run_id", "Artefactos anteriores sin informe"),
        modo_reentrenamiento="Manual por API o scripts; sin tarea automática configurada",
        mensaje=("Último intento fallido: " + status.get("error", "Error de entrenamiento")) if state == "FALLIDO"
                else "El proceso anterior se interrumpió. Puedes iniciar otro entrenamiento." if state == "INTERRUMPIDO"
                else "Entrenamiento de XGBoost y comparación medidos sobre los CSV sintéticos actuales.",
    )


def ejecutar_reentrenamiento() -> PlanificadorReentrenamientoResponse:
    train_models()
    return obtener_planificador_reentrenamiento()
