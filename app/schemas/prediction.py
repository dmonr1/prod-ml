from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PrediccionGlobalRequest(BaseModel):
    matricula_id: int
    periodo_evaluacion_id: Optional[int] = None
    corte_seguimiento_id: Optional[int] = None
    semana_corte: Optional[int] = None
    fecha_corte: Optional[date] = None
    promedio_general: float = Field(ge=0, le=20)
    cantidad_cursos: int = Field(ge=1)
    cantidad_cursos_desaprobados: int = Field(ge=0)
    nota_maxima: float = Field(ge=0, le=20)
    nota_minima: float = Field(ge=0, le=20)
    clases_programadas: int = Field(ge=0)
    clases_asistidas: int = Field(ge=0)
    porcentaje_asistencia: float = Field(ge=0, le=100)
    cantidad_evaluaciones_registradas: int = Field(default=0, ge=0)
    cantidad_notas_desaprobadas_total: int = Field(default=0, ge=0)
    cantidad_notas_criticas_total: int = Field(default=0, ge=0)
    peor_nota_periodo: float = Field(default=20, ge=0, le=20)
    cantidad_cursos_c: int = Field(default=0, ge=0)
    cantidad_cursos_b: int = Field(default=0, ge=0)
    cantidad_cursos_a: int = Field(default=0, ge=0)
    cantidad_cursos_ad: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validar_metadatos_corte(self):
        if self.corte_seguimiento_id is not None and (self.semana_corte is None or self.fecha_corte is None):
            raise ValueError("Un corte de seguimiento requiere semana y fecha de corte")
        return self


class PrediccionCursoRequest(BaseModel):
    matricula_id: int
    curso_id: int
    curso_nombre: str
    periodo_evaluacion_id: Optional[int] = None
    corte_seguimiento_id: Optional[int] = None
    semana_corte: Optional[int] = None
    fecha_corte: Optional[date] = None
    nota_curso: float = Field(ge=0, le=20)
    promedio_general: float = Field(ge=0, le=20)
    cantidad_cursos_desaprobados: int = Field(ge=0)
    porcentaje_asistencia: float = Field(ge=0, le=100)
    cantidad_evaluaciones_registradas: int = Field(default=0, ge=0)
    nota_minima_curso: float = Field(default=20, ge=0, le=20)
    nota_maxima_curso: float = Field(default=20, ge=0, le=20)
    cantidad_notas_desaprobadas: int = Field(default=0, ge=0)
    cantidad_notas_criticas: int = Field(default=0, ge=0)
    nota_examen_principal: float = Field(default=20, ge=0, le=20)
    cantidad_notas_c: int = Field(default=0, ge=0)
    cantidad_notas_b: int = Field(default=0, ge=0)
    cantidad_notas_a: int = Field(default=0, ge=0)
    cantidad_notas_ad: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validar_metadatos_corte(self):
        if self.corte_seguimiento_id is not None and (self.semana_corte is None or self.fecha_corte is None):
            raise ValueError("Un corte de seguimiento requiere semana y fecha de corte")
        return self


class PredictRequest(BaseModel):
    modelo_version: str = "v1"
    global_features: PrediccionGlobalRequest
    course_features: List[PrediccionCursoRequest]

    @model_validator(mode="after")
    def validar_consistencia_corte(self):
        corte_id = self.global_features.corte_seguimiento_id
        if corte_id is not None and any(item.corte_seguimiento_id != corte_id for item in self.course_features):
            raise ValueError("Las predicciones de curso deben pertenecer al mismo corte global")
        return self


class PrediccionGlobalResponse(BaseModel):
    matricula_id: int
    periodo_evaluacion_id: Optional[int] = None
    corte_seguimiento_id: Optional[int] = None
    semana_corte: Optional[int] = None
    fecha_corte: Optional[date] = None
    puntaje_riesgo: float
    nivel_riesgo: str
    modelo_version: str
    variables_entrada: dict


class PrediccionCursoResponse(BaseModel):
    matricula_id: int
    curso_id: int
    curso_nombre: str
    periodo_evaluacion_id: Optional[int] = None
    corte_seguimiento_id: Optional[int] = None
    semana_corte: Optional[int] = None
    fecha_corte: Optional[date] = None
    puntaje_riesgo: float
    nivel_riesgo: str
    modelo_version: str
    variables_entrada: dict


class PredictResponse(BaseModel):
    global_prediction: PrediccionGlobalResponse
    course_predictions: List[PrediccionCursoResponse]
    pattern_summary: Optional[str] = None
