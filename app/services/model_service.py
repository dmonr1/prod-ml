import joblib
import pandas as pd

from app.config import COURSE_MODEL_PATH, GLOBAL_MODEL_PATH
from app.config import RISK_THRESHOLDS
from app.schemas.prediction import (
    PrediccionCursoRequest,
    PrediccionCursoResponse,
    PrediccionGlobalRequest,
    PrediccionGlobalResponse,
)
from app.schemas.training import FEATURES_CURSO, FEATURES_GLOBALES
from app.services.feature_service import preparar_features_curso, preparar_features_globales

_global_model = None
_global_label_encoder = None
_course_model = None
_course_label_encoder = None


def _clasificar_riesgo(puntaje: float) -> str:
    if puntaje <= RISK_THRESHOLDS["bajo_max"]:
        return "BAJO"
    if puntaje <= RISK_THRESHOLDS["medio_max"]:
        return "MEDIO"
    return "ALTO"


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 100.0) -> float:
    return round(max(minimo, min(maximo, valor)), 2)


def _cargar_modelo_global():
    global _global_model, _global_label_encoder

    if _global_model is None and GLOBAL_MODEL_PATH.exists():
        _global_model = joblib.load(GLOBAL_MODEL_PATH)

    encoder_path = COURSE_MODEL_PATH.parent / "label_encoder_global.joblib"
    if _global_label_encoder is None and encoder_path.exists():
        _global_label_encoder = joblib.load(encoder_path)

    return _global_model, _global_label_encoder


def _cargar_modelo_curso():
    global _course_model, _course_label_encoder

    if _course_model is None and COURSE_MODEL_PATH.exists():
        _course_model = joblib.load(COURSE_MODEL_PATH)

    encoder_path = COURSE_MODEL_PATH.parent / "label_encoder_curso.joblib"
    if _course_label_encoder is None and encoder_path.exists():
        _course_label_encoder = joblib.load(encoder_path)

    return _course_model, _course_label_encoder


def predecir_riesgo_global(payload: PrediccionGlobalRequest, modelo_version: str) -> PrediccionGlobalResponse:
    features = preparar_features_globales(payload)

    model, label_encoder = _cargar_modelo_global()

    if model is not None and label_encoder is not None:
        try:
            dataframe = pd.DataFrame([[features[col] for col in FEATURES_GLOBALES]], columns=FEATURES_GLOBALES)
            probabilidades = model.predict_proba(dataframe)[0]
            indice_predicho = int(probabilidades.argmax())
            nivel = str(label_encoder.inverse_transform([indice_predicho])[0])
            puntaje = _limitar(float(probabilidades[indice_predicho]) * 100.0)
        except Exception:
            puntaje, nivel = _predecir_global_heuristico(payload)
    else:
        puntaje, nivel = _predecir_global_heuristico(payload)

    return PrediccionGlobalResponse(
        matricula_id=payload.matricula_id,
        bimestre_id=payload.bimestre_id,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_global_heuristico(payload: PrediccionGlobalRequest) -> tuple[float, str]:
        # Fallback heuristico temporal si el modelo no esta disponible.
        riesgo = 0.0
        riesgo += max(0.0, (14 - payload.promedio_general) * 8)
        riesgo += payload.cantidad_cursos_desaprobados * 10
        riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.7)
        riesgo += max(0.0, (11 - payload.nota_minima) * 6)
        if payload.cantidad_evaluaciones_registradas <= 1:
            riesgo += 4
        puntaje = _limitar(riesgo)
        nivel = _clasificar_riesgo(puntaje)
        return puntaje, nivel


def predecir_riesgo_curso(payload: PrediccionCursoRequest, modelo_version: str) -> PrediccionCursoResponse:
    features = preparar_features_curso(payload)

    model, label_encoder = _cargar_modelo_curso()

    if model is not None and label_encoder is not None:
        try:
            dataframe = pd.DataFrame([[features[col] for col in FEATURES_CURSO]], columns=FEATURES_CURSO)
            probabilidades = model.predict_proba(dataframe)[0]
            indice_predicho = int(probabilidades.argmax())
            nivel = str(label_encoder.inverse_transform([indice_predicho])[0])
            puntaje = _limitar(float(probabilidades[indice_predicho]) * 100.0)
        except Exception:
            puntaje, nivel = _predecir_curso_heuristico(payload)
    else:
        puntaje, nivel = _predecir_curso_heuristico(payload)

    return PrediccionCursoResponse(
        matricula_id=payload.matricula_id,
        curso_id=payload.curso_id,
        curso_nombre=payload.curso_nombre,
        bimestre_id=payload.bimestre_id,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_curso_heuristico(payload: PrediccionCursoRequest) -> tuple[float, str]:
        riesgo = 0.0
        riesgo += max(0.0, (14 - payload.nota_curso) * 9)
        riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.5)
        riesgo += payload.cantidad_cursos_desaprobados * 4
        riesgo += max(0.0, (13 - payload.promedio_general) * 5)
        if payload.cantidad_evaluaciones_registradas <= 1:
            riesgo += 3
        puntaje = _limitar(riesgo)
        nivel = _clasificar_riesgo(puntaje)
        return puntaje, nivel
