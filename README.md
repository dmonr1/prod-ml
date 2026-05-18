# Rendimiento Academico ML

Servicio Python para el modulo de Data Mining y Machine Learning del sistema de rendimiento academico.

## Objetivo

Este proyecto expone una API para:

- recibir variables academicas desde Spring Boot
- aplicar transformaciones y analisis con PCA
- ejecutar predicciones de riesgo de fracaso academico con XGBoost
- devolver probabilidad de fracaso global y por curso

## Estructura

```text
app/
  main.py
  config.py
  schemas/
  services/
data/
trained_models/
scripts/
```

## Endpoint principal

`POST /predict`

Recibe:

- una prediccion global por alumno
- varias predicciones por curso del mismo alumno

Devuelve:

- probabilidad de fracaso y nivel de riesgo global
- probabilidad de fracaso y nivel de riesgo por curso

## Levantar en local

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Estado actual

La base del servicio ya esta creada.

Por ahora:

- la API esta operativa
- existe modelo binario para probabilidad de fracaso global
- existe modelo binario para probabilidad de fracaso por curso
- existen datasets iniciales de entrenamiento para riesgo global y por curso
- existen scripts de entrenamiento con PCA + XGBoost

## Entrenamiento inicial

Dataset base:

- [data/dataset_riesgo_global.csv](data/dataset_riesgo_global.csv)
- [data/dataset_riesgo_curso.csv](data/dataset_riesgo_curso.csv)

Script:

```bash
python -m scripts.train_global_model
python -m scripts.train_course_model
```

Salida esperada:

- `trained_models/modelo_riesgo_global.joblib`
- `trained_models/pca_transformer.joblib`
- `trained_models/label_encoder_global.joblib`
- `trained_models/modelo_riesgo_curso.joblib`
- `trained_models/pca_transformer_curso.joblib`
- `trained_models/label_encoder_curso.joblib`
