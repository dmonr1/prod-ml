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
]

TARGET_GLOBAL = "riesgo_global"

FEATURES_CURSO: List[str] = [
    "nota_curso",
    "promedio_general",
    "porcentaje_asistencia",
    "cantidad_evaluaciones_registradas",
]

TARGET_CURSO = "riesgo_curso"
