import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from app.config import DATA_DIR
from app.schemas.predictors import (
    PredictorFeatureDto,
    PredictorsConfigResponse,
)

CONFIG_FILE = DATA_DIR / "predictors_config.json"

DEFAULT_GLOBAL_PREDICTORS = [
    {
        "key": "promedio_general",
        "label": "Promedio General Acumulado",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Calificación media global obtenida por el estudiante en todas sus asignaturas.",
        "activo": True,
        "peso": 2.5,
    },
    {
        "key": "porcentaje_asistencia",
        "label": "Porcentaje de Asistencia Global",
        "tipo": "GLOBAL",
        "categoria": "ASISTENCIA",
        "descripcion": "Tasa efectiva de sesiones asistidas frente al total de clases programadas.",
        "activo": True,
        "peso": 2.0,
    },
    {
        "key": "nota_minima",
        "label": "Nota Mínima del Período",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Menor calificación individual registrada en cualquier evaluación del período.",
        "activo": True,
        "peso": 1.8,
    },
    {
        "key": "cantidad_cursos_desaprobados",
        "label": "Cursos Desaprobados",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Número total de asignaturas con promedio proyectado menor a 11.",
        "activo": True,
        "peso": 2.2,
    },
    {
        "key": "cantidad_evaluaciones_registradas",
        "label": "Evaluaciones Registradas",
        "tipo": "GLOBAL",
        "categoria": "EVALUATIVO",
        "descripcion": "Volumen de evidencias calificadas acumuladas a la fecha de corte.",
        "activo": True,
        "peso": 1.0,
    },
    {
        "key": "cantidad_notas_criticas_total",
        "label": "Calificaciones Críticas (<=07)",
        "tipo": "GLOBAL",
        "categoria": "EVALUATIVO",
        "descripcion": "Cantidad de evaluaciones con calificaciones severamente deficientes.",
        "activo": True,
        "peso": 2.0,
    },
    {
        "key": "cantidad_notas_desaprobadas_total",
        "label": "Calificaciones Desaprobadas (<=10)",
        "tipo": "GLOBAL",
        "categoria": "EVALUATIVO",
        "descripcion": "Frecuencia de notas registradas por debajo del estándar de aprobación.",
        "activo": True,
        "peso": 1.6,
    },
    {
        "key": "clases_asistidas",
        "label": "Sesiones Asistidas",
        "tipo": "GLOBAL",
        "categoria": "ASISTENCIA",
        "descripcion": "Conteo total de asistencias presenciales o síncronas efectivas.",
        "activo": True,
        "peso": 1.0,
    },
    {
        "key": "clases_programadas",
        "label": "Sesiones Programadas",
        "tipo": "GLOBAL",
        "categoria": "ASISTENCIA",
        "descripcion": "Carga horaria lectiva impartida acumulada hasta la fecha.",
        "activo": True,
        "peso": 0.8,
    },
    {
        "key": "peor_nota_periodo",
        "label": "Peor Nota en Evaluaciones Principales",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Menor puntuación en evaluaciones de alto peso académico.",
        "activo": True,
        "peso": 1.5,
    },
    {
        "key": "nota_maxima",
        "label": "Nota Máxima Obtenida",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Rendimiento pico del alumno como indicador de potencial recuperador.",
        "activo": True,
        "peso": 0.7,
    },
    {
        "key": "cantidad_cursos",
        "label": "Carga de Cursos Matriculados",
        "tipo": "GLOBAL",
        "categoria": "ACADEMICO",
        "descripcion": "Número total de asignaturas activas cursadas simultáneamente.",
        "activo": True,
        "peso": 0.6,
    },
]

DEFAULT_COURSE_PREDICTORS = [
    {
        "key": "nota_curso",
        "label": "Promedio Actual del Curso",
        "tipo": "CURSO",
        "categoria": "ACADEMICO",
        "descripcion": "Calificación media ponderada del alumno en la asignatura específica.",
        "activo": True,
        "peso": 2.8,
    },
    {
        "key": "porcentaje_asistencia",
        "label": "Asistencia Específica al Curso",
        "tipo": "CURSO",
        "categoria": "ASISTENCIA",
        "descripcion": "Porcentaje de concurrencia a las sesiones programadas de esta materia.",
        "activo": True,
        "peso": 2.1,
    },
    {
        "key": "nota_minima_curso",
        "label": "Nota Mínima Registrada en Curso",
        "tipo": "CURSO",
        "categoria": "ACADEMICO",
        "descripcion": "Puntuación más baja alcanzada en las evaluaciones de la asignatura.",
        "activo": True,
        "peso": 1.9,
    },
    {
        "key": "promedio_general",
        "label": "Contexto de Promedio General",
        "tipo": "CURSO",
        "categoria": "ACADEMICO",
        "descripcion": "Rendimiento holístico del estudiante en su trayectoria del período.",
        "activo": True,
        "peso": 1.4,
    },
    {
        "key": "cantidad_evaluaciones_registradas",
        "label": "Evaluaciones Calificadas en Curso",
        "tipo": "CURSO",
        "categoria": "EVALUATIVO",
        "descripcion": "Número de calificaciones registradas por el docente titular.",
        "activo": True,
        "peso": 1.2,
    },
    {
        "key": "nota_examen_principal",
        "label": "Nota en Examen Parcial o Principal",
        "tipo": "CURSO",
        "categoria": "EVALUATIVO",
        "descripcion": "Calificación obtenida en la evaluación sumativa de mayor ponderación.",
        "activo": True,
        "peso": 2.4,
    },
    {
        "key": "cantidad_notas_criticas",
        "label": "Notas Críticas en Curso (<=07)",
        "tipo": "CURSO",
        "categoria": "EVALUATIVO",
        "descripcion": "Frecuencia de evaluaciones del curso con nota inferior a 08.",
        "activo": True,
        "peso": 2.0,
    },
    {
        "key": "cantidad_notas_desaprobadas",
        "label": "Notas Desaprobadas en Curso (<=10)",
        "tipo": "CURSO",
        "categoria": "EVALUATIVO",
        "descripcion": "Frecuencia de evaluaciones del curso por debajo de 11.",
        "activo": True,
        "peso": 1.7,
    },
    {
        "key": "nota_maxima_curso",
        "label": "Nota Máxima en Curso",
        "tipo": "CURSO",
        "categoria": "ACADEMICO",
        "descripcion": "Mejor calificación obtenida por el alumno dentro de la asignatura.",
        "activo": True,
        "peso": 0.8,
    },
    {
        "key": "cantidad_cursos_desaprobados",
        "label": "Cursos Paralelos en Riesgo",
        "tipo": "CURSO",
        "categoria": "ACADEMICO",
        "descripcion": "Impacto colateral de otras materias desaprobadas en el desempeño del curso.",
        "activo": True,
        "peso": 1.1,
    },
]


def _asegurar_configuracion_inicial() -> Dict[str, list]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        datos = {
            "global_features": DEFAULT_GLOBAL_PREDICTORS,
            "course_features": DEFAULT_COURSE_PREDICTORS,
            "ultima_actualizacion": datetime.now().isoformat(),
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2, ensure_ascii=False)
        return datos

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "global_features": DEFAULT_GLOBAL_PREDICTORS,
            "course_features": DEFAULT_COURSE_PREDICTORS,
            "ultima_actualizacion": datetime.now().isoformat(),
        }


def obtener_configuracion_predictores() -> PredictorsConfigResponse:
    datos = _asegurar_configuracion_inicial()
    global_items = [PredictorFeatureDto(**item) for item in datos.get("global_features", [])]
    course_items = [PredictorFeatureDto(**item) for item in datos.get("course_features", [])]
    total_activos = sum(1 for item in global_items + course_items if item.activo)

    return PredictorsConfigResponse(
        global_features=global_items,
        course_features=course_items,
        total_activos=total_activos,
        ultima_actualizacion=datos.get("ultima_actualizacion"),
    )


def guardar_configuracion_predictores(features: List[PredictorFeatureDto]) -> PredictorsConfigResponse:
    global_items = [f.model_dump() for f in features if f.tipo == "GLOBAL"]
    course_items = [f.model_dump() for f in features if f.tipo == "CURSO"]

    # Si se actualizó solo una sección, preservar los otros del archivo existente
    datos_actuales = _asegurar_configuracion_inicial()
    if not global_items:
        global_items = datos_actuales.get("global_features", DEFAULT_GLOBAL_PREDICTORS)
    if not course_items:
        course_items = datos_actuales.get("course_features", DEFAULT_COURSE_PREDICTORS)

    ahora = datetime.now().isoformat()
    datos_guardar = {
        "global_features": global_items,
        "course_features": course_items,
        "ultima_actualizacion": ahora,
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(datos_guardar, f, indent=2, ensure_ascii=False)

    return obtener_configuracion_predictores()


def obtener_features_globales_activas() -> List[str]:
    cfg = obtener_configuracion_predictores()
    return [f.key for f in cfg.global_features if f.activo]


def obtener_features_curso_activas() -> List[str]:
    cfg = obtener_configuracion_predictores()
    return [f.key for f in cfg.course_features if f.activo]
