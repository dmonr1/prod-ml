from fastapi import FastAPI

from app.config import COURSE_MODEL_PATH, GLOBAL_MODEL_PATH, MODEL_TRAINING_SOURCE
from app.schemas.prediction import PredictRequest, PredictResponse
from app.schemas.predictors import (
    ComparativaModelosResponse,
    PlanificadorReentrenamientoResponse,
    PredictorsConfigResponse,
    PredictorsConfigUpdateRequest,
)
from app.services.model_comparison_service import (
    ejecutar_reentrenamiento_simulado,
    obtener_comparativa_modelos,
    obtener_planificador_reentrenamiento,
)
from app.services.model_service import predecir_riesgo_curso, predecir_riesgo_global
from app.services.pattern_service import generar_resumen_patrones
from app.services.predictor_service import (
    guardar_configuracion_predictores,
    obtener_configuracion_predictores,
)

app = FastAPI(
    title="Rendimiento Academico ML",
    version="1.0.0",
    description="Servicio Python para prediccion de riesgo academico global y por curso, comparativa multialgoritmo y configuracion de predictores",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "rendimiento-academico-ml",
        "model_training_source": MODEL_TRAINING_SOURCE,
        "global_model_available": GLOBAL_MODEL_PATH.exists(),
        "course_model_available": COURSE_MODEL_PATH.exists(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    global_prediction = predecir_riesgo_global(payload.global_features, payload.modelo_version)
    course_predictions = [
        predecir_riesgo_curso(course_feature, payload.modelo_version)
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
    return guardar_configuracion_predictores(payload.features)


@app.get("/models/comparison", response_model=ComparativaModelosResponse)
def get_models_comparison() -> ComparativaModelosResponse:
    return obtener_comparativa_modelos()


@app.get("/retraining/schedule", response_model=PlanificadorReentrenamientoResponse)
def get_retraining_schedule() -> PlanificadorReentrenamientoResponse:
    return obtener_planificador_reentrenamiento()


@app.post("/retraining/run", response_model=PlanificadorReentrenamientoResponse)
def run_retraining() -> PlanificadorReentrenamientoResponse:
    return ejecutar_reentrenamiento_simulado()
