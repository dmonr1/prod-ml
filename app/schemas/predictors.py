from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class PredictorFeatureDto(BaseModel):
    key: str
    label: str
    tipo: Literal["GLOBAL", "CURSO"]
    categoria: Literal["ACADEMICO", "ASISTENCIA", "EVALUATIVO"]
    descripcion: str
    activo: bool = True
    peso: float = Field(default=1.0, ge=0.0, le=5.0)


class PredictorsConfigResponse(BaseModel):
    global_features: List[PredictorFeatureDto]
    course_features: List[PredictorFeatureDto]
    total_activos: int
    ultima_actualizacion: Optional[str] = None


class PredictorsConfigUpdateRequest(BaseModel):
    features: List[PredictorFeatureDto]


class AlgoritmoComparativaDto(BaseModel):
    id: str
    nombre: str
    familia: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    latencia_ms: float
    estado: Literal["ACTIVO", "CANDIDATO", "BASELINE"]
    ranking: int
    hiperparametros: dict


class ComparativaModelosResponse(BaseModel):
    algoritmos: List[AlgoritmoComparativaDto]
    modelo_recomendado: str
    metrica_optimizada: str
    fecha_evaluacion: str
    total_registros_evaluados: int


class PlanificadorReentrenamientoResponse(BaseModel):
    cadencia: str
    proxima_ejecucion_programada: str
    ultimo_reentrenamiento: str
    estado_ultimo_reentrenamiento: str
    registros_entrenamiento: int
    modelo_actual_version: str
    modo_reentrenamiento: str
    mensaje: Optional[str] = None
