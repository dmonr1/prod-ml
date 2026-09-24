from fastapi import FastAPI

from app.config import COURSE_MODEL_PATH, GLOBAL_MODEL_PATH, MODEL_TRAINING_SOURCE
from app.schemas.prediction import PredictRequest, PredictResponse
from app.services.model_service import predecir_riesgo_curso, predecir_riesgo_global
from app.services.pattern_service import generar_resumen_patrones

app = FastAPI(
    title="Rendimiento Academico ML",
    version="1.0.0",
    description="Servicio Python para prediccion de riesgo academico global y por curso",
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
