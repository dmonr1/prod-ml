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


def _ajustar_puntaje_global(payload: PrediccionGlobalRequest, puntaje: float) -> float:
    ajuste = 0.0

    promedio_solido = payload.promedio_general >= 13.0
    asistencia_alta = payload.porcentaje_asistencia >= 90
    sin_fracaso_real = payload.cantidad_cursos_desaprobados == 0
    cursos_fragiles_controlados = payload.cantidad_cursos_c <= 1 and payload.cantidad_cursos_b <= 3
    bloque_estable = payload.cantidad_cursos_a + payload.cantidad_cursos_ad >= max(1, payload.cantidad_cursos // 2)
    criticidad_acotada = payload.cantidad_notas_criticas_total <= 3
    peor_nota_controlada = payload.peor_nota_periodo >= 8

    if promedio_solido and asistencia_alta and sin_fracaso_real and cursos_fragiles_controlados and bloque_estable and criticidad_acotada and peor_nota_controlada:
        ajuste -= 24
    elif payload.promedio_general >= 12.5 and payload.porcentaje_asistencia >= 85 and payload.cantidad_cursos_desaprobados == 0:
        ajuste -= 12

    if payload.peor_nota_periodo <= 5:
        ajuste += 7
    elif payload.peor_nota_periodo <= 7:
        ajuste += 3

    if payload.cantidad_notas_criticas_total >= 4:
        ajuste += 6
    elif payload.cantidad_notas_criticas_total == 3:
        ajuste += 3

    if payload.cantidad_cursos_c >= 3:
        ajuste += 8
    elif payload.cantidad_cursos_c == 2:
        ajuste += 4
    elif payload.cantidad_cursos_c == 1:
        ajuste += 1

    puntaje_ajustado = _limitar(puntaje + ajuste)

    if promedio_solido and asistencia_alta and sin_fracaso_real and cursos_fragiles_controlados and bloque_estable and criticidad_acotada and peor_nota_controlada:
        puntaje_ajustado = min(puntaje_ajustado, 58.0)

    return puntaje_ajustado


def predecir_riesgo_global(payload: PrediccionGlobalRequest, modelo_version: str) -> PrediccionGlobalResponse:
    features = preparar_features_globales(payload)

    model = _cargar_modelo_global()
    dataframe = pd.DataFrame([[features[col] for col in FEATURES_GLOBALES]], columns=FEATURES_GLOBALES)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_global_heuristico(payload)
    else:
        puntaje = _ajustar_puntaje_global(payload, puntaje)
        puntaje = _acotar_probabilidad_visible(puntaje)
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
    riesgo = 0.0
    riesgo += max(0.0, (11 - payload.promedio_general) * 12)
    riesgo += payload.cantidad_cursos_desaprobados * 16
    riesgo += payload.cantidad_notas_desaprobadas_total * 2.5
    riesgo += payload.cantidad_notas_criticas_total * 4.5
    riesgo += payload.cantidad_cursos_c * 14
    riesgo += payload.cantidad_cursos_b * 4
    riesgo -= payload.cantidad_cursos_a * 2
    riesgo -= payload.cantidad_cursos_ad * 4
    riesgo += max(0.0, (11 - payload.nota_minima) * 10)
    riesgo += max(0.0, (11 - payload.peor_nota_periodo) * 8)
    riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.25)
    if payload.cantidad_evaluaciones_registradas <= 1:
        riesgo += 5
    puntaje = _acotar_probabilidad_visible(_limitar(riesgo))
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel


def _ajustar_puntaje_curso(payload: PrediccionCursoRequest, puntaje: float) -> float:
    ajuste = 0.0

    promedio_solido = payload.nota_curso >= 13.0
    examen_aprobado = payload.nota_examen_principal >= 13.0
    asistencia_alta = payload.porcentaje_asistencia >= 90
    fragilidad_acotada = payload.cantidad_notas_desaprobadas <= 3 and payload.cantidad_notas_criticas <= 3
    bloque_fuerte = payload.cantidad_notas_a + payload.cantidad_notas_ad >= 5

    if promedio_solido and examen_aprobado and asistencia_alta and fragilidad_acotada and bloque_fuerte:
        ajuste -= 22
    elif payload.nota_curso >= 12.0 and payload.nota_examen_principal >= 11.0 and payload.porcentaje_asistencia >= 85:
        ajuste -= 10

    if payload.nota_minima_curso <= 5:
        ajuste += 6
    elif payload.nota_minima_curso <= 8:
        ajuste += 3

    if payload.cantidad_notas_criticas >= 3:
        ajuste += 6
    elif payload.cantidad_notas_criticas == 2:
        ajuste += 2

    if payload.cantidad_notas_c >= 4:
        ajuste += 7
    elif payload.cantidad_notas_c == 3:
        ajuste += 3
    elif payload.cantidad_notas_c == 2:
        ajuste += 1

    if payload.nota_examen_principal < 11:
        ajuste += 5

    puntaje_ajustado = _limitar(puntaje + ajuste)

    if promedio_solido and examen_aprobado and asistencia_alta and fragilidad_acotada and bloque_fuerte:
        puntaje_ajustado = min(puntaje_ajustado, 56.0)

    return puntaje_ajustado


def predecir_riesgo_curso(payload: PrediccionCursoRequest, modelo_version: str) -> PrediccionCursoResponse:
    features = preparar_features_curso(payload)

    model = _cargar_modelo_curso()
    dataframe = pd.DataFrame([[features[col] for col in FEATURES_CURSO]], columns=FEATURES_CURSO)
    puntaje = _obtener_probabilidad_clase_positiva(model, dataframe)

    if puntaje is None:
        puntaje, nivel = _predecir_curso_heuristico(payload)
    else:
        puntaje = _ajustar_puntaje_curso(payload, puntaje)
        puntaje = _acotar_probabilidad_visible(puntaje)
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
    riesgo += max(0.0, (11 - payload.nota_curso) * 14)
    riesgo += max(0.0, (11 - payload.nota_minima_curso) * 10)
    riesgo += payload.cantidad_notas_desaprobadas * 6
    riesgo += payload.cantidad_notas_criticas * 9
    riesgo += payload.cantidad_notas_c * 8
    riesgo += payload.cantidad_notas_b * 2
    riesgo -= payload.cantidad_notas_a * 1.5
    riesgo -= payload.cantidad_notas_ad * 2.5
    riesgo += max(0.0, (11 - payload.nota_examen_principal) * 7)
    riesgo += max(0.0, (11 - payload.promedio_general) * 6)
    riesgo += max(0.0, (85 - payload.porcentaje_asistencia) * 0.2)
    if payload.cantidad_evaluaciones_registradas <= 1:
        riesgo += 4
    puntaje = _acotar_probabilidad_visible(_limitar(riesgo))
    nivel = _clasificar_riesgo(puntaje)
    return puntaje, nivel


