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
        "cantidad_notas_desaprobadas_total": payload.cantidad_notas_desaprobadas_total,
        "cantidad_notas_criticas_total": payload.cantidad_notas_criticas_total,
        "peor_nota_periodo": payload.peor_nota_periodo,
        "cantidad_cursos_c": payload.cantidad_cursos_c,
        "cantidad_cursos_b": payload.cantidad_cursos_b,
        "cantidad_cursos_a": payload.cantidad_cursos_a,
        "cantidad_cursos_ad": payload.cantidad_cursos_ad,
    }


def preparar_features_curso(payload: PrediccionCursoRequest) -> dict:
    return {
        "nota_curso": payload.nota_curso,
        "promedio_general": payload.promedio_general,
        "porcentaje_asistencia": payload.porcentaje_asistencia,
        "cantidad_evaluaciones_registradas": payload.cantidad_evaluaciones_registradas,
        "nota_minima_curso": payload.nota_minima_curso,
        "nota_maxima_curso": payload.nota_maxima_curso,
        "cantidad_notas_desaprobadas": payload.cantidad_notas_desaprobadas,
        "cantidad_notas_criticas": payload.cantidad_notas_criticas,
        "nota_examen_principal": payload.nota_examen_principal,
        "cantidad_notas_c": payload.cantidad_notas_c,
        "cantidad_notas_b": payload.cantidad_notas_b,
        "cantidad_notas_a": payload.cantidad_notas_a,
        "cantidad_notas_ad": payload.cantidad_notas_ad,
    }
