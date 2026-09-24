# Rendimiento Academico ML

Servicio FastAPI para recibir indicadores academicos desde Spring Boot y calcular riesgo global y por curso.

## Estado de los modelos

Los artefactos activos se entrenan con XGBoost usando el conjunto sintetico de cortes descrito abajo. No representan un modelo validado con historiales reales del colegio.

Los CSV iniciales no incluyen fecha de corte y contienen variables agregadas del mismo periodo que define el resultado. Por ello no permiten medir una alerta temprana sin fuga de informacion. No generar snapshots inventados a partir de promedios finales.

## Contrato de entrenamiento

Cada fila debe representar la informacion disponible para un alumno antes del resultado final. Se requieren estas columnas:

- `alumno_id`: identifica al alumno para mantener sus registros en una sola particion.
- `fecha_corte`: fecha hasta la cual se calculan las variables predictoras.
- `fecha_resultado`: fecha posterior en que se determina el resultado final.
- `fracaso_global` o `fracaso_curso`: etiqueta final binaria, segun el modelo.

Las fechas se validan para garantizar `fecha_corte < fecha_resultado`. El conjunto de prueba se separa por alumno y los predictores se limitan a indicadores parciales disponibles al corte. Los scripts imprimen Accuracy, Precision, Recall, F1, AUC y matriz de confusion; solo guardan los artefactos si el contrato y la particion son validos.

Columnas predictoras requeridas:

- Global: `promedio_general`, `cantidad_cursos`, `nota_maxima`, `nota_minima`, `clases_programadas`, `clases_asistidas`, `porcentaje_asistencia`, `cantidad_evaluaciones_registradas`.
- Por curso: `nota_curso`, `promedio_general`, `porcentaje_asistencia`, `cantidad_evaluaciones_registradas`, `nota_minima_curso`, `nota_maxima_curso`.

Los objetivos son `fracaso_global` y `fracaso_curso`, calculados con el resultado final posterior al corte, nunca con esas variables parciales.

## Entrenar con historial real

Para sustituir los modelos sinteticos por modelos validos para el colegio, reemplaza estos CSV con snapshots retrospectivos fechados, derivados de registros academicos reales:

- [data/dataset_riesgo_global.csv](data/dataset_riesgo_global.csv)
- [data/dataset_riesgo_curso.csv](data/dataset_riesgo_curso.csv)

Luego ejecuta los entrenadores de datos reales:

```bash
python -m scripts.train_global_model
python -m scripts.train_course_model
```

## Datos sinteticos y entrenamiento activo

La base local disponible no contiene suficientes historiales fechados para construir cortes reales. Se puede generar un conjunto reproducible de datos sinteticos y entrenar con el:

```bash
python -m scripts.generate_synthetic_cutoff_datasets
python -m scripts.train_synthetic_demo
```

Por defecto genera 2,000 alumnos simulados, cuatro fechas de corte por alumno, 8,000 filas globales y 40,000 filas curso-alumno. Los CSV quedan en `data/synthetic_demo/` y se marcan con `origen_datos=SIMULADO_SOLO_PARA_PRUEBAS`. El entrenamiento reemplaza los artefactos activos en `trained_models/`, que son los que carga la API. La API reporta `synthetic_cutoff_simulation` en `/health`. Las metricas y probabilidades no estan validadas con alumnos reales ni son evidencia del Colegio San Marcos. Aunque estos modelos se pueden desplegar para demostracion, sus puntajes no deben usarse para decisiones academicas ni presentarse como resultados reales en la tesis.

## Ejecutar la API

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

El endpoint de inferencia es `POST /predict`.
