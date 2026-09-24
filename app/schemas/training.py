from typing import List


FEATURES_GLOBALES: List[str] = [
    "promedio_general",
    "cantidad_cursos",
    "nota_maxima",
    "nota_minima",
    "clases_programadas",
    "clases_asistidas",
    "porcentaje_asistencia",
    "cantidad_evaluaciones_registradas",
    "cantidad_notas_desaprobadas_total",
    "cantidad_notas_criticas_total",
    "cantidad_cursos_desaprobados",
    "peor_nota_periodo",
    "cantidad_cursos_c",
    "cantidad_cursos_b",
    "cantidad_cursos_a",
    "cantidad_cursos_ad",
]

FEATURES_GLOBALES_CORTE: List[str] = [
    "promedio_general",
    "cantidad_cursos",
    "nota_maxima",
    "nota_minima",
    "clases_programadas",
    "clases_asistidas",
    "porcentaje_asistencia",
    "cantidad_evaluaciones_registradas",
]

TARGET_GLOBAL = "fracaso_global"

FEATURES_CURSO: List[str] = [
    "nota_curso",
    "promedio_general",
    "porcentaje_asistencia",
    "cantidad_evaluaciones_registradas",
    "nota_minima_curso",
    "nota_maxima_curso",
    "cantidad_notas_desaprobadas",
    "cantidad_notas_criticas",
    "nota_examen_principal",
    "cantidad_notas_c",
    "cantidad_notas_b",
    "cantidad_notas_a",
    "cantidad_notas_ad",
]

FEATURES_CURSO_CORTE: List[str] = [
    "nota_curso",
    "promedio_general",
    "porcentaje_asistencia",
    "cantidad_evaluaciones_registradas",
    "nota_minima_curso",
    "nota_maxima_curso",
]

TARGET_CURSO = "fracaso_curso"
