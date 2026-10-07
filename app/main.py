import logging
from typing import Literal
from fastapi import FastAPI, HTTPException

from app.config import MODEL_TRAINING_SOURCE
from app.schemas.prediction import PredictRequest, PredictResponse
from app.schemas.predictors import (
    ComparativaModelosResponse,
    PlanificadorReentrenamientoResponse,
    PredictorsConfigResponse,
    PredictorsConfigUpdateRequest,
)
from app.services.model_comparison_service import (
    ejecutar_reentrenamiento,
    obtener_comparativa_modelos,
    obtener_planificador_reentrenamiento,
)
from app.services.model_service import predecir_riesgo_curso, predecir_riesgo_global
from app.services.artifact_service import active_run, active_model_path, TrainingInProgressError
from app.services.pattern_service import generar_resumen_patrones
from app.services.predictor_service import (
    obtener_configuracion_predictores,
)

app = FastAPI(
    title="Rendimiento Academico ML",
    version="1.0.0",
    description="Servicio Python para prediccion de riesgo academico global y por curso, comparativa multialgoritmo y configuracion de predictores",
)


@app.get("/health")
def health() -> dict:
    manifest = active_run()
    return {
        "status": "ok",
        "service": "rendimiento-academico-ml",
        "model_training_source": MODEL_TRAINING_SOURCE,
        "global_model_available": active_model_path("global", manifest).exists(),
        "course_model_available": active_model_path("course", manifest).exists(),
        "active_run": manifest.get("run_id"),
        "global_features": manifest.get("models", {}).get("global", {}).get("features", []),
        "course_features": manifest.get("models", {}).get("course", {}).get("features", []),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    manifest = active_run()
    global_prediction = predecir_riesgo_global(payload.global_features, payload.modelo_version, manifest)
    course_predictions = [
        predecir_riesgo_curso(course_feature, payload.modelo_version, manifest)
        for course_feature in payload.course_features
    ]
    pattern_summary = generar_resumen_patrones(payload)

    return PredictResponse(
        global_prediction=global_prediction,
        course_predictions=course_predictions,
        pattern_summary=pattern_summary,
    )


@app.get("/predictors/config", response_model=PredictorsConfigResponse)
def get_predictors_config() -> PredictorsConfigResponse:
    return obtener_configuracion_predictores()


@app.put("/predictors/config", response_model=PredictorsConfigResponse)
def update_predictors_config(payload: PredictorsConfigUpdateRequest) -> PredictorsConfigResponse:
    raise HTTPException(status_code=409, detail="Los predictores se obtienen del modelo entrenado y no son editables desde esta pantalla.")


@app.get("/models/comparison", response_model=ComparativaModelosResponse)
def get_models_comparison(task: Literal["global", "course"] = "global") -> ComparativaModelosResponse:
    try:
        return obtener_comparativa_modelos(task)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/retraining/schedule", response_model=PlanificadorReentrenamientoResponse)
def get_retraining_schedule() -> PlanificadorReentrenamientoResponse:
    return obtener_planificador_reentrenamiento()


@app.post("/retraining/run", response_model=PlanificadorReentrenamientoResponse)
def run_retraining() -> PlanificadorReentrenamientoResponse:
    try:
        return ejecutar_reentrenamiento()
    except TrainingInProgressError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Error de reentrenamiento")
        raise HTTPException(status_code=500, detail="Falló el entrenamiento; consulta el estado de la última ejecución.") from exc
