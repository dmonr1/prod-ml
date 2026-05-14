from typing import List, Optional

from pydantic import BaseModel, Field


class PrediccionGlobalRequest(BaseModel):
    matricula_id: int
    periodo_evaluacion_id: int
    promedio_general: float = Field(ge=0, le=20)
    cantidad_cursos: int = Field(ge=1)
    cantidad_cursos_desaprobados: int = Field(ge=0)
    nota_maxima: float = Field(ge=0, le=20)
    nota_minima: float = Field(ge=0, le=20)
    clases_programadas: int = Field(ge=0)
    clases_asistidas: int = Field(ge=0)
    porcentaje_asistencia: float = Field(ge=0, le=100)
    cantidad_evaluaciones_registradas: int = Field(default=0, ge=0)


class PrediccionCursoRequest(BaseModel):
    matricula_id: int
    curso_id: int
    curso_nombre: str
    periodo_evaluacion_id: int
    nota_curso: float = Field(ge=0, le=20)
    promedio_general: float = Field(ge=0, le=20)
    cantidad_cursos_desaprobados: int = Field(ge=0)
    porcentaje_asistencia: float = Field(ge=0, le=100)
    cantidad_evaluaciones_registradas: int = Field(default=0, ge=0)


class PredictRequest(BaseModel):
    modelo_version: str = "v1"
    global_features: PrediccionGlobalRequest
    course_features: List[PrediccionCursoRequest]


class PrediccionGlobalResponse(BaseModel):
    matricula_id: int
    periodo_evaluacion_id: int
    puntaje_riesgo: float
    nivel_riesgo: str
    modelo_version: str
    variables_entrada: dict


class PrediccionCursoResponse(BaseModel):
    matricula_id: int
    curso_id: int
    curso_nombre: str
    periodo_evaluacion_id: int
    puntaje_riesgo: float
    nivel_riesgo: str
    modelo_version: str
    variables_entrada: dict


class PredictResponse(BaseModel):
    global_prediction: PrediccionGlobalResponse
    course_predictions: List[PrediccionCursoResponse]
    pattern_summary: Optional[str] = None
