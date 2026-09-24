"""Generate explicitly synthetic temporal datasets for pipeline testing only."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "synthetic_demo"
CUTOFFS = pd.to_datetime(["2025-05-15", "2025-07-27", "2025-10-08", "2025-12-10"])
RESULT_DATE = pd.Timestamp("2025-12-20")
COURSE_COUNT = 5


def generate(students: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    global_rows = []
    course_rows = []

    for student_id in range(1, students + 1):
        ability = rng.normal(12.8, 2.6)
        attendance_habit = np.clip(rng.beta(8, 2), 0.35, 1.0)
        course_difficulties = rng.normal(0, 2.0, COURSE_COUNT)
        final_grades = np.clip(ability - course_difficulties + rng.normal(0, 1.5, COURSE_COUNT), 0, 20)
        global_failure = int(bool(np.any(final_grades < 11) or np.mean(final_grades) < 11))

        for cutoff_index, cutoff in enumerate(CUTOFFS):
            progress = (cutoff_index + 1) / len(CUTOFFS)
            scheduled_per_course = (cutoff_index + 1) * 10
            attended_per_course = int(round(scheduled_per_course * np.clip(attendance_habit + rng.normal(0, 0.035), 0, 1)))
            attendance_pct = 100.0 * attended_per_course / scheduled_per_course
            registered_per_course = (cutoff_index + 1) * 3

            partial_grades = np.clip(
                11.0 + progress * (final_grades - 11.0) + rng.normal(0, 1.8, COURSE_COUNT),
                0,
                20,
            )
            global_average = float(np.mean(partial_grades))

            global_rows.append({
                "alumno_id": f"SIM-{student_id:05d}",
                "fecha_corte": cutoff.date().isoformat(),
                "fecha_resultado": RESULT_DATE.date().isoformat(),
                "promedio_general": round(global_average, 2),
                "cantidad_cursos": COURSE_COUNT,
                "nota_maxima": round(float(np.max(partial_grades)), 2),
                "nota_minima": round(float(np.min(partial_grades)), 2),
                "clases_programadas": scheduled_per_course * COURSE_COUNT,
                "clases_asistidas": attended_per_course * COURSE_COUNT,
                "porcentaje_asistencia": round(attendance_pct, 2),
                "cantidad_evaluaciones_registradas": registered_per_course * COURSE_COUNT,
                "fracaso_global": global_failure,
                "origen_datos": "SIMULADO_SOLO_PARA_PRUEBAS",
            })

            for course_index, partial_grade in enumerate(partial_grades):
                final_course_grade = final_grades[course_index]
                course_rows.append({
                    "alumno_id": f"SIM-{student_id:05d}",
                    "curso_id": course_index + 1,
                    "fecha_corte": cutoff.date().isoformat(),
                    "fecha_resultado": RESULT_DATE.date().isoformat(),
                    "nota_curso": round(float(partial_grade), 2),
                    "promedio_general": round(global_average, 2),
                    "porcentaje_asistencia": round(attendance_pct, 2),
                    "cantidad_evaluaciones_registradas": registered_per_course,
                    "nota_minima_curso": round(max(0.0, float(partial_grade - rng.uniform(0, 2.5))), 2),
                    "nota_maxima_curso": round(min(20.0, float(partial_grade + rng.uniform(0, 2.5))), 2),
                    "fracaso_curso": int(final_course_grade < 11),
                    "origen_datos": "SIMULADO_SOLO_PARA_PRUEBAS",
                })

    return pd.DataFrame(global_rows), pd.DataFrame(course_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--students", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.students < 100:
        parser.error("Usa al menos 100 alumnos para probar particiones agrupadas y validación cruzada.")

    global_data, course_data = generate(args.students, args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    global_path = OUTPUT_DIR / "dataset_riesgo_global_sintetico.csv"
    course_path = OUTPUT_DIR / "dataset_riesgo_curso_sintetico.csv"
    global_data.to_csv(global_path, index=False)
    course_data.to_csv(course_path, index=False)
    print(f"Global: {len(global_data):,} filas, {global_data['alumno_id'].nunique():,} alumnos -> {global_path}")
    print(f"Curso: {len(course_data):,} filas, {course_data['alumno_id'].nunique():,} alumnos -> {course_path}")
    print("ADVERTENCIA: datos completamente sintéticos; no usar métricas en la tesis ni como evidencia institucional.")


if __name__ == "__main__":
    main()
