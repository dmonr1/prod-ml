"""Describe the predictors of the fitted models, not an independent UI configuration."""
from app.schemas.predictors import PredictorFeatureDto, PredictorsConfigResponse
from app.schemas.training import FEATURES_CURSO_CORTE, FEATURES_GLOBALES_CORTE
from app.services.artifact_service import active_run, model_report

DESCRIPTIONS = {
    "promedio_general": ("Promedio general al corte", "Media de las notas disponibles de los cursos al corte."),
    "cantidad_cursos": ("Cursos con información", "Cantidad de cursos incluidos en el cálculo al corte."),
    "nota_maxima": ("Mayor promedio al corte", "Mayor promedio de curso disponible."),
    "nota_minima": ("Menor promedio al corte", "Menor promedio de curso disponible."),
    "clases_programadas": ("Clases programadas", "Sesiones acumuladas hasta el corte."),
    "clases_asistidas": ("Clases asistidas", "Sesiones asistidas acumuladas hasta el corte."),
    "porcentaje_asistencia": ("Asistencia al corte", "Porcentaje de sesiones asistidas."),
    "cantidad_evaluaciones_registradas": ("Evaluaciones registradas", "Cantidad de evaluaciones con nota al corte."),
    "nota_curso": ("Promedio del curso", "Promedio de las notas del curso disponibles al corte."),
    "nota_minima_curso": ("Nota mínima del curso", "Menor nota del curso disponible al corte."),
    "nota_maxima_curso": ("Nota máxima del curso", "Mayor nota del curso disponible al corte."),
}


def obtener_configuracion_predictores() -> PredictorsConfigResponse:
    manifest = active_run()
    results = {}
    for task, kind, default in (("global", "GLOBAL", FEATURES_GLOBALES_CORTE), ("course", "CURSO", FEATURES_CURSO_CORTE)):
        try:
            report = model_report(task)
        except FileNotFoundError:
            report = {}
        features = report.get("features", default)
        results[task] = []
        for key in features:
            label, description = DESCRIPTIONS[key]
            category = "ASISTENCIA" if key in {"clases_programadas", "clases_asistidas", "porcentaje_asistencia"} else "EVALUATIVO" if key == "cantidad_evaluaciones_registradas" else "ACADEMICO"
            results[task].append(PredictorFeatureDto(
                key=key, label=label, tipo=kind, categoria=category, descripcion=description,
                activo=bool(report), peso=report.get("feature_importance_gain", {}).get(key, 0.0),
            ))
    return PredictorsConfigResponse(
        global_features=results["global"], course_features=results["course"],
        total_activos=sum(item.activo for items in results.values() for item in items),
        ultima_actualizacion=manifest.get("completed_at"),
    )


def guardar_configuracion_predictores(features):
    raise ValueError("Los predictores pertenecen al contrato del modelo entrenado. Su modificación requiere cambiar el contrato y reentrenar.")
