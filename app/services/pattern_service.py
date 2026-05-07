from app.schemas.prediction import PredictRequest


def generar_resumen_patrones(payload: PredictRequest) -> str:
    asistencia = payload.global_features.porcentaje_asistencia
    desaprobados = payload.global_features.cantidad_cursos_desaprobados
    evaluaciones = payload.global_features.cantidad_evaluaciones_registradas
    cursos_riesgo_alto = [
        course.curso_nombre
        for course in payload.course_features
        if course.nota_curso < 11
    ]

    if asistencia < 80 and desaprobados >= 2:
        return "Patron detectado: baja asistencia y multiples cursos desaprobados."
    if cursos_riesgo_alto:
        return (
            "Patron detectado: existen cursos con desempeno critico ("
            + ", ".join(cursos_riesgo_alto[:3])
            + ")."
        )
    if evaluaciones <= 1:
        return "Patron detectado: aun hay pocas evaluaciones registradas para una prediccion mas estable."
    if asistencia < 85:
        return "Patron detectado: la asistencia podria estar impactando el rendimiento."
    if desaprobados >= 2:
        return "Patron detectado: el estudiante presenta riesgo por acumulacion de cursos desaprobados."
    return "Sin patrones criticos detectados en esta evaluacion."
