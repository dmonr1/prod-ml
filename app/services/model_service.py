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
from app.schemas.training import FEATURES_CURSO, FEATURES_CURSO_CORTE, FEATURES_GLOBALES, FEATURES_GLOBALES_CORTE
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

def _acotar_probabilidad_visible(puntaje: float) -> float:
    if puntaje <= 0.0:
        return 3.0
    if puntaje >= 100.0:
        return 97.0
    if puntaje < 5.0:
        return 5.0
    if puntaje > 95.0:
        return 95.0
    return round(puntaje, 2)


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


def _nombres_features_del_modelo(model, fallback: list[str]) -> list[str]:
    if model is None:
        return fallback
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    return [str(name) for name in names] if names is not None else fallback


def _ajustar_puntaje_global(payload: PrediccionGlobalRequest, puntaje: float) -> float:
    ajuste = 0.0

    promedio_solido = payload.promedio_general >= 13.0
    asistencia_alta = payload.porcentaje_asistencia >= 90
    notas_parciales_controladas = payload.nota_minima >= 8
    seguimiento_suficiente = payload.cantidad_evaluaciones_registradas >= 2

    if promedio_solido and asistencia_alta and notas_parciales_controladas and seguimiento_suficiente:
        ajuste -= 24
    elif payload.promedio_general >= 12.5 and payload.porcentaje_asistencia >= 85:
        ajuste -= 12

    if payload.nota_minima <= 5:
        ajuste += 7
    elif payload.nota_minima <= 7:
        ajuste += 3

    puntaje_ajustado = _limitar(puntaje + ajuste)

    if promedio_solido and asistencia_alta and notas_parciales_controladas and seguimiento_suficiente:
        puntaje_ajustado = min(puntaje_ajustado, 58.0)

    return puntaje_ajustado


def predecir_riesgo_global(payload: PrediccionGlobalRequest, modelo_version: str) -> PrediccionGlobalResponse:
    features = preparar_features_globales(payload)

    model = _cargar_modelo_global()
    expected_features = FEATURES_GLOBALES_CORTE if payload.corte_seguimiento_id is not None else FEATURES_GLOBALES
    model_features = _nombres_features_del_modelo(model, expected_features)
    if payload.corte_seguimiento_id is not None and model_features != FEATURES_GLOBALES_CORTE:
        model = None
        model_features = FEATURES_GLOBALES_CORTE
    dataframe = pd.DataFrame([[features[col] for col in model_features]], columns=model_features)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_global_heuristico(payload)
    else:
        puntaje = _ajustar_puntaje_global(payload, puntaje)
        nivel = _clasificar_riesgo(puntaje)

    return PrediccionGlobalResponse(
        matricula_id=payload.matricula_id,
        periodo_evaluacion_id=payload.periodo_evaluacion_id,
        corte_seguimiento_id=payload.corte_seguimiento_id,
        semana_corte=payload.semana_corte,
        fecha_corte=payload.fecha_corte,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_global_heuristico(payload: PrediccionGlobalRequest) -> tuple[float, str]:
    riesgo = 0.0
    riesgo += max(0.0, (11 - payload.promedio_general) * 12)
    riesgo += max(0.0, (11 - payload.nota_minima) * 10)
    if payload.clases_programadas > 0:
        riesgo += max(0.0, (90 - payload.porcentaje_asistencia) * 0.35)
    if payload.cantidad_evaluaciones_registradas < 2:
        riesgo += 3
    puntaje = _acotar_probabilidad_visible(_limitar(riesgo))
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel


def _ajustar_puntaje_curso(payload: PrediccionCursoRequest, puntaje: float) -> float:
    ajuste = 0.0

    promedio_solido = payload.nota_curso >= 13.0
    asistencia_alta = payload.porcentaje_asistencia >= 90
    notas_parciales_controladas = payload.nota_minima_curso >= 8
    seguimiento_suficiente = payload.cantidad_evaluaciones_registradas >= 2

    if promedio_solido and asistencia_alta and notas_parciales_controladas and seguimiento_suficiente:
        ajuste -= 22
    elif payload.nota_curso >= 12.0 and payload.porcentaje_asistencia >= 85:
        ajuste -= 10

    if payload.nota_minima_curso <= 5:
        ajuste += 6
    elif payload.nota_minima_curso <= 8:
        ajuste += 3

    puntaje_ajustado = _limitar(puntaje + ajuste)

    if promedio_solido and asistencia_alta and notas_parciales_controladas and seguimiento_suficiente:
        puntaje_ajustado = min(puntaje_ajustado, 56.0)

    return puntaje_ajustado


def predecir_riesgo_curso(payload: PrediccionCursoRequest, modelo_version: str) -> PrediccionCursoResponse:
    features = preparar_features_curso(payload)

    model = _cargar_modelo_curso()
    expected_features = FEATURES_CURSO_CORTE if payload.corte_seguimiento_id is not None else FEATURES_CURSO
    model_features = _nombres_features_del_modelo(model, expected_features)
    if payload.corte_seguimiento_id is not None and model_features != FEATURES_CURSO_CORTE:
        model = None
        model_features = FEATURES_CURSO_CORTE
    dataframe = pd.DataFrame([[features[col] for col in model_features]], columns=model_features)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_curso_heuristico(payload)
    else:
        puntaje = _ajustar_puntaje_curso(payload, puntaje)
        nivel = _clasificar_riesgo(puntaje)

    return PrediccionCursoResponse(
        matricula_id=payload.matricula_id,
        curso_id=payload.curso_id,
        curso_nombre=payload.curso_nombre,
        periodo_evaluacion_id=payload.periodo_evaluacion_id,
        corte_seguimiento_id=payload.corte_seguimiento_id,
        semana_corte=payload.semana_corte,
        fecha_corte=payload.fecha_corte,
        puntaje_riesgo=puntaje,
        nivel_riesgo=nivel,
        modelo_version=modelo_version,
        variables_entrada=features,
    )


def _predecir_curso_heuristico(payload: PrediccionCursoRequest) -> tuple[float, str]:
    riesgo = 0.0
    riesgo += max(0.0, (11 - payload.nota_curso) * 14)
    riesgo += max(0.0, (11 - payload.nota_minima_curso) * 10)
    riesgo += max(0.0, (11 - payload.promedio_general) * 6)
    if payload.porcentaje_asistencia > 0:
        riesgo += max(0.0, (90 - payload.porcentaje_asistencia) * 0.3)
    if payload.cantidad_evaluaciones_registradas < 2:
        riesgo += 3
    puntaje = _acotar_probabilidad_visible(_limitar(riesgo))
    
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel


 
