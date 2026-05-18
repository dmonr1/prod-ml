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
_course_model = None


def _clasificar_riesgo(puntaje: float) -> str:
    if puntaje <= RISK_THRESHOLDS["bajo_max"]:
        return "BAJO"
    if puntaje <= RISK_THRESHOLDS["medio_max"]:
        return "MEDIO"
    return "ALTO"


def _limitar(valor: float, minimo: float = 0.0, maximo: float = 100.0) -> float:
    return round(max(minimo, min(maximo, valor)), 2)


def _cargar_modelo_global():
    global _global_model

    if _global_model is None and GLOBAL_MODEL_PATH.exists():
        _global_model = joblib.load(GLOBAL_MODEL_PATH)

    return _global_model


def _cargar_modelo_curso():
    global _course_model

    if _course_model is None and COURSE_MODEL_PATH.exists():
        _course_model = joblib.load(COURSE_MODEL_PATH)

    return _course_model


def _obtener_probabilidad_clase_positiva(model, dataframe: pd.DataFrame) -> float | None:
    if model is None or not hasattr(model, "predict_proba"):
        return None

    try:
        probabilidades = model.predict_proba(dataframe)[0]
        classes = getattr(model, "classes_", None)

        if classes is None and hasattr(model, "named_steps"):
            predictor = model.named_steps.get("model")
            classes = getattr(predictor, "classes_", None)

        if classes is None:
            return None

        classes_list = [int(value) for value in list(classes)]
        if 1 not in classes_list or len(classes_list) != 2:
            return None

        indice_positivo = classes_list.index(1)
        return _limitar(float(probabilidades[indice_positivo]) * 100.0)
    except Exception:
        return None


def predecir_riesgo_global(payload: PrediccionGlobalRequest, modelo_version: str) -> PrediccionGlobalResponse:
    features = preparar_features_globales(payload)

    model = _cargar_modelo_global()
    dataframe = pd.DataFrame([[features[col] for col in FEATURES_GLOBALES]], columns=FEATURES_GLOBALES)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_global_heuristico(payload)
    else:
        nivel = _clasificar_riesgo(puntaje)

    return PrediccionGlobalResponse(
        matricula_id=payload.matricula_id,
        periodo_evaluacion_id=payload.periodo_evaluacion_id,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_global_heuristico(payload: PrediccionGlobalRequest) -> tuple[float, str]:
    # Riesgo de fracaso academico global estimado como probabilidad 0-100.
    riesgo = 0.0
    riesgo += max(0.0, (11 - payload.promedio_general) * 12)
    riesgo += payload.cantidad_cursos_desaprobados * 18
    riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.45)
    riesgo += max(0.0, (11 - payload.nota_minima) * 10)
    if payload.cantidad_evaluaciones_registradas <= 1:
        riesgo += 5
    puntaje = _limitar(riesgo)
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel


def predecir_riesgo_curso(payload: PrediccionCursoRequest, modelo_version: str) -> PrediccionCursoResponse:
    features = preparar_features_curso(payload)

    model = _cargar_modelo_curso()
    dataframe = pd.DataFrame([[features[col] for col in FEATURES_CURSO]], columns=FEATURES_CURSO)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_curso_heuristico(payload)
    else:
        nivel = _clasificar_riesgo(puntaje)

    return PrediccionCursoResponse(
        matricula_id=payload.matricula_id,
        curso_id=payload.curso_id,
        curso_nombre=payload.curso_nombre,
        periodo_evaluacion_id=payload.periodo_evaluacion_id,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_curso_heuristico(payload: PrediccionCursoRequest) -> tuple[float, str]:
    riesgo = 0.0
    riesgo += max(0.0, (11 - payload.nota_curso) * 16)
    riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.35)
    riesgo += payload.cantidad_cursos_desaprobados * 6
    riesgo += max(0.0, (11 - payload.promedio_general) * 8)
    if payload.cantidad_evaluaciones_registradas <= 1:
        riesgo += 4
    puntaje = _limitar(riesgo)
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel
