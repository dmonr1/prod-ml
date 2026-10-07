# Rendimiento Académico ML

FastAPI recibe indicadores desde Spring Boot y calcula riesgo global y por curso.
Los datos activos son **sintéticos**, generados para probar el sistema; no son
historiales del colegio ni validación de resultados educativos.

## Fuente de datos única

- `data/dataset_riesgo_global.csv`: 8.000 filas de 2.000 alumnos simulados.
- `data/dataset_riesgo_curso.csv`: 40.000 filas; cinco cursos y cuatro cortes por alumno.
- Cada fila incluye `alumno_id`, `fecha_corte` y `fecha_resultado`.
- La procedencia sintética se registra en `MODEL_TRAINING_SOURCE` y en los
  informes JSON del entrenamiento; los CSV activos no incluyen `origen_datos`.
- Los CSV anteriores se mantienen fuera del repositorio porque su procedencia no está verificada.
- `data/synthetic_demo/` es una copia histórica; ya no alimenta entrenadores.

Variables globales: `promedio_general`, `cantidad_cursos`, `nota_maxima`,
`nota_minima`, `clases_programadas`, `clases_asistidas`, `porcentaje_asistencia`,
`cantidad_evaluaciones_registradas`.

Variables por curso: `nota_curso`, `promedio_general`, `porcentaje_asistencia`,
`cantidad_evaluaciones_registradas`, `nota_minima_curso`, `nota_maxima_curso`.

Etiquetas: `fracaso_global` y `fracaso_curso`. En el generador, el fracaso de
curso significa nota final menor a 11; el global, algún curso con nota final
menor a 11 o promedio final menor a 11. Las variables parciales son simulaciones
condicionadas a esas notas finales. Su separación por fechas permite probar el
contrato de datos, pero no acredita capacidad de anticipación en el colegio.

## Entrenar y evaluar

Desde la raíz, con las dependencias de requirements.txt:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_synthetic_cutoff_datasets
.\.venv\Scripts\python.exe -m scripts.train_synthetic_demo
```

El primer comando regenera los CSV canónicos (semilla 42, 2.000 alumnos).
El segundo entrena ambos modelos y calcula las comparaciones. También se puede
actualizar solo uno con `python -m scripts.train_global_model` o
`python -m scripts.train_course_model`. Todos usan el mismo contrato y las mismas
rutas. La procedencia se conserva en los metadatos de cada ejecución. Incorporar
historial real requiere revisar su procedencia y protocolo.

La prueba reserva el 25% de alumnos con GroupShuffleSplit y semilla 42. No se
mezclan alumnos entre entrenamiento y prueba. La validación cruzada de XGBoost
usa cinco pliegues StratifiedGroupKFold **solo en entrenamiento** y reporta media
y desviación estándar muestral (ddof=1). El umbral binario es 0,5. Los pesos de
clase se calculan en cada partición de entrenamiento. No hay selección de
hiperparámetros ni selección automática del modelo desplegado usando la prueba.

XGBoost, Random Forest, Gradient Boosting y regresión logística se comparan en
la misma partición de cada tarea. StandardScaler se ajusta solo en entrenamiento
para regresión logística. El modelo activo sigue siendo XGBoost aunque otro
algoritmo obtenga mayor F1. Los informes corresponden al clasificador crudo;
los ajustes heurísticos posteriores del servicio de predicción no están evaluados
por esas métricas. El tiempo reportado es por fila de un lote, no latencia HTTP.

## Artefactos y API

Cada ejecución guarda modelos, informes JSON y predicciones de prueba CSV en
`trained_models/runs/<version>/`. `trained_models/active_run.json` selecciona la
generación activa mediante reemplazo atómico. La API detecta la versión nueva
sin reiniciarse. Si falla un entrenamiento, se conserva la generación anterior.
Los `.joblib` de la raíz son artefactos anteriores utilizados únicamente cuando
no hay manifiesto; no son el destino de nuevos entrenamientos.

- `GET /health`: origen, versión y variables actuales.
- `GET /models/comparison?task=global` o `task=course`: métricas guardadas.
- `GET /predictors/config`: las 8 y 6 variables, con importancia por ganancia
  del modelo de evaluación. Esta vista es de consulta; los pesos de la antigua
  pantalla no modificaban el clasificador y ya no se presentan como editables.
- `GET /retraining/schedule`: estado persistido, sin inventar una tarea semanal.
- `POST /retraining/run`: ejecuta y evalúa ambos modelos; retorna al finalizar.
  Devuelve error si falla, y HTTP 409 si hay otra ejecución activa.
- `POST /predict`: inferencia con la versión activa. Un respaldo heurístico se
  identifica como `heuristic-fallback`, sin atribuirlo al modelo entrenado.

El bloqueo de entrenamiento coordina procesos CLI y API sobre el mismo disco.
No hay planificador automático. Para despliegues con varios contenedores se
necesita almacenamiento compartido y coordinación externa. Conservar el directorio
trained_models en almacenamiento persistente si se reentrena en Railway.

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Los tests HTTP requieren httpx (requirements-dev.txt). Los informes y predicciones
de evaluación quedan disponibles para documentar resultados reproducibles con su
origen sintético explícito.
