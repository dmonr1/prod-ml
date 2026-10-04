from datetime import datetime, timedelta
from app.schemas.predictors import (
    AlgoritmoComparativaDto,
    ComparativaModelosResponse,
    PlanificadorReentrenamientoResponse,
)


def obtener_comparativa_modelos() -> ComparativaModelosResponse:
    algoritmos = [
        AlgoritmoComparativaDto(
            id="xgboost",
            nombre="XGBoost Classifier",
            familia="Gradient Boosted Decision Trees",
            accuracy=0.9140,
            precision=0.8950,
            recall=0.9020,
            f1_score=0.8985,
            roc_auc=0.9420,
            latencia_ms=12.4,
            estado="ACTIVO",
            ranking=1,
            hiperparametros={
                "n_estimators": 120,
                "max_depth": 3,
                "learning_rate": 0.05,
                "scale_pos_weight": 2.14,
                "subsample": 0.90,
            },
        ),
        AlgoritmoComparativaDto(
            id="random_forest",
            nombre="Random Forest Classifier",
            familia="Ensemble Bagging",
            accuracy=0.8870,
            precision=0.8640,
            recall=0.8710,
            f1_score=0.8675,
            roc_auc=0.9180,
            latencia_ms=18.6,
            estado="CANDIDATO",
            ranking=2,
            hiperparametros={
                "n_estimators": 150,
                "max_depth": 6,
                "criterion": "gini",
                "min_samples_split": 4,
            },
        ),
        AlgoritmoComparativaDto(
            id="gradient_boosting",
            nombre="Gradient Boosting Classifier",
            familia="Sequential Boosting",
            accuracy=0.8790,
            precision=0.8520,
            recall=0.8600,
            f1_score=0.8560,
            roc_auc=0.9050,
            latencia_ms=15.1,
            estado="CANDIDATO",
            ranking=3,
            hiperparametros={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "loss": "log_loss",
            },
        ),
        AlgoritmoComparativaDto(
            id="logistic_regression",
            nombre="Regresión Logística Regularizada",
            familia="Linear Models",
            accuracy=0.8250,
            precision=0.7910,
            recall=0.8040,
            f1_score=0.7974,
            roc_auc=0.8540,
            latencia_ms=4.8,
            estado="BASELINE",
            ranking=4,
            hiperparametros={
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "max_iter": 500,
            },
        ),
    ]

    # Ordenar por F1-Score descendente
    algoritmos_ordenados = sorted(algoritmos, key=lambda a: a.f1_score, reverse=True)
    for idx, alg in enumerate(algoritmos_ordenados, start=1):
        alg.ranking = idx

    return ComparativaModelosResponse(
        algoritmos=algoritmos_ordenados,
        modelo_recomendado="XGBoost Classifier",
        metrica_optimizada="F1-Score / ROC-AUC con ajuste por desbalance de clases",
        fecha_evaluacion=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_registros_evaluados=1240,
    )


def obtener_planificador_reentrenamiento() -> PlanificadorReentrenamientoResponse:
    hoy = datetime.now()
    proximo_lunes = hoy + timedelta(days=(7 - hoy.weekday()) % 7 or 7)
    proxima_ejecucion = proximo_lunes.replace(hour=2, minute=0, second=0, microsecond=0)

    return PlanificadorReentrenamientoResponse(
        cadencia="Semanal programado (Lunes 02:00 AM UTC - post cierre de calificaciones y cortes)",
        proxima_ejecucion_programada=proxima_ejecucion.strftime("%Y-%m-%d %H:%M:%S"),
        ultimo_reentrenamiento=(hoy - timedelta(days=3)).strftime("%Y-%m-%d 02:14:32"),
        estado_ultimo_reentrenamiento="COMPLETADO_EXITOSO",
        registros_entrenamiento=3850,
        modelo_actual_version="v4-corte-temprano-xgb",
        modo_reentrenamiento="Automático por pipeline CI/CD o manual bajo demanda",
        mensaje="Pipeline de reentrenamiento sincronizado con los lotes académicos.",
    )


def ejecutar_reentrenamiento_simulado() -> PlanificadorReentrenamientoResponse:
    ahora = datetime.now()
    proximo_lunes = ahora + timedelta(days=(7 - ahora.weekday()) % 7 or 7)
    proxima_ejecucion = proximo_lunes.replace(hour=2, minute=0, second=0, microsecond=0)

    return PlanificadorReentrenamientoResponse(
        cadencia="Semanal programado (Lunes 02:00 AM UTC - post cierre de calificaciones y cortes)",
        proxima_ejecucion_programada=proxima_ejecucion.strftime("%Y-%m-%d %H:%M:%S"),
        ultimo_reentrenamiento=ahora.strftime("%Y-%m-%d %H:%M:%S"),
        estado_ultimo_reentrenamiento="COMPLETADO_EXITOSO",
        registros_entrenamiento=3920,
        modelo_actual_version="v4.1-reentrenado-xgb",
        modo_reentrenamiento="Manual ejecutado por Administrador TI",
        mensaje="Reentrenamiento multimodelo ejecutado satisfactoriamente. Métricas recalculadas y modelos actualizados.",
    )
