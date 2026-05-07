from app.schemas.prediction import PrediccionCursoRequest, PrediccionGlobalRequest


def preparar_features_globales(payload: PrediccionGlobalRequest) -> dict:
    return {
        "promedio_general": payload.promedio_general,
        "cantidad_cursos": payload.cantidad_cursos,
        "cantidad_cursos_desaprobados": payload.cantidad_cursos_desaprobados,
        "nota_maxima": payload.nota_maxima,
        "nota_minima": payload.nota_minima,
        "clases_programadas": payload.clases_programadas,
        "clases_asistidas": payload.clases_asistidas,
        "porcentaje_asistencia": payload.porcentaje_asistencia,
        "cantidad_evaluaciones_registradas": payload.cantidad_evaluaciones_registradas,
    }


def preparar_features_curso(payload: PrediccionCursoRequest) -> dict:
    return {
        "nota_curso": payload.nota_curso,
        "promedio_general": payload.promedio_general,
        "cantidad_cursos_desaprobados": payload.cantidad_cursos_desaprobados,
        "porcentaje_asistencia": payload.porcentaje_asistencia,
        "cantidad_evaluaciones_registradas": payload.cantidad_evaluaciones_registradas,
    }
