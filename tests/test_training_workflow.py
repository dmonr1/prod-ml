import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.schemas.training import FEATURES_CURSO_CORTE, FEATURES_GLOBALES_CORTE
from app.services.artifact_service import active_run, active_model_path, model_report, training_lock
from app.services.model_service import _load_task
from app.services.training_service import train_models
from scripts.generate_synthetic_cutoff_datasets import generate
from scripts.training_common import load_dataset, metrics


class TrainingWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.patches = patch.multiple(config,
            TRAINED_MODELS_DIR=root / 'models', ACTIVE_RUN_PATH=root / 'models/active_run.json',
            TRAINING_STATUS_PATH=root / 'models/training_status.json',
            GLOBAL_DATASET_PATH=root / 'global.csv', COURSE_DATASET_PATH=root / 'course.csv')
        cls.patches.start()
        cls.global_data, cls.course_data = generate(120, 42)
        cls.global_data.to_csv(config.GLOBAL_DATASET_PATH, index=False)
        cls.course_data.to_csv(config.COURSE_DATASET_PATH, index=False)
        cls.client = TestClient(app)
        response = cls.client.post('/retraining/run')
        if response.status_code != 200:
            raise AssertionError(response.text)

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.patches.stop()
        cls.temp.cleanup()

    def test_metrics_recompute_from_saved_predictions_and_students_are_disjoint(self):
        for task in ('global', 'course'):
            report = model_report(task)
            training, testing = set(report['train_students']), set(report['test_students'])
            self.assertFalse(training & testing)
            for fold in report['cv']['folds']:
                tr, va = set(fold['train_students']), set(fold['validation_students'])
                self.assertFalse(tr & va)
                self.assertEqual(training, tr | va)
                self.assertFalse(testing & (tr | va))
            predictions = pd.read_csv(active_model_path(task).with_suffix('.predictions.csv'))
            for algorithm in report['algorithms']:
                observed = metrics(predictions[report['target']], predictions[algorithm['id'] + '_probability'].to_numpy())
                self.assertAlmostEqual(observed['f1_score'], algorithm['f1_score'])
                self.assertAlmostEqual(observed['roc_auc'], algorithm['roc_auc'])
            response = self.client.get('/models/comparison', params={'task': task})
            self.assertEqual(200, response.status_code)
            self.assertEqual(report['test_rows'], response.json()['total_registros_evaluados'])
            self.assertEqual('synthetic_cutoff_simulation', response.json()['origen_datos'])

    def test_failures_never_publish_partial_models_or_report_success(self):
        before = config.ACTIVE_RUN_PATH.read_bytes()
        with patch('app.services.training_service.entrenar_modelo_temporal',
                   side_effect=[{'rows': len(self.global_data)}, ValueError('CSV de curso incompatible')]):
            response = self.client.post('/retraining/run')
        self.assertEqual(422, response.status_code)
        self.assertEqual(before, config.ACTIVE_RUN_PATH.read_bytes())
        self.assertEqual('FALLIDO', self.client.get('/retraining/schedule').json()['estado_ultimo_reentrenamiento'])
        with training_lock():
            response = self.client.post('/retraining/run')
            self.assertEqual('EN_EJECUCION', self.client.get('/retraining/schedule').json()['estado_ultimo_reentrenamiento'])
        self.assertEqual(409, response.status_code)
        self.assertEqual(before, config.ACTIVE_RUN_PATH.read_bytes())

    def test_interrupted_process_does_not_leave_the_button_permanently_locked(self):
        previous = config.TRAINING_STATUS_PATH.read_bytes()
        try:
            config.TRAINING_STATUS_PATH.write_text(json.dumps({'status': 'EN_EJECUCION'}), encoding='utf-8')
            self.assertEqual('INTERRUMPIDO', self.client.get('/retraining/schedule').json()['estado_ultimo_reentrenamiento'])
        finally:
            config.TRAINING_STATUS_PATH.write_bytes(previous)

    def test_updated_model_is_loaded_in_same_process_and_inference_reports_its_version(self):
        old = _load_task('global')
        result = train_models(('global',))
        self.assertIsNot(old, _load_task('global'))
        g = self.global_data.iloc[0].to_dict()
        g.update(matricula_id=1, cantidad_cursos_desaprobados=0, corte_seguimiento_id=1, semana_corte=1)
        c = self.course_data.iloc[0].to_dict()
        c.update(matricula_id=1, cantidad_cursos_desaprobados=0, curso_nombre='Curso', corte_seguimiento_id=1, semana_corte=1)
        response = self.client.post('/predict', json={'global_features': g, 'course_features': [c]})
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual(result['models']['global']['run_id'], response.json()['global_prediction']['modelo_version'])
        self.assertEqual(result['models']['course']['run_id'], response.json()['course_predictions'][0]['modelo_version'])
        self.assertEqual(8, len(self.client.get('/predictors/config').json()['global_features']))
        self.assertEqual(6, len(self.client.get('/predictors/config').json()['course_features']))

    def test_legacy_csv_invalid_labels_and_temporal_leakage_are_rejected(self):
        variants = [self.global_data.drop(columns=['fecha_corte']),
                    self.global_data.assign(fracaso_global=0.5),
                    self.global_data.assign(fecha_corte='2026-01-01'),
                    self.global_data.assign(origen_datos='REAL')]
        for index, frame in enumerate(variants):
            path = Path(self.temp.name) / f'invalid-{index}.csv'
            frame.to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_dataset(path, 'fracaso_global', FEATURES_GLOBALES_CORTE)

    def test_missing_evaluation_and_invalid_task_do_not_return_demo_metrics(self):
        with patch('app.services.model_comparison_service.model_report', side_effect=FileNotFoundError('Sin evaluación')):
            self.assertEqual(409, self.client.get('/models/comparison').status_code)
        self.assertEqual(422, self.client.get('/models/comparison?task=unknown').status_code)
        self.assertEqual(409, self.client.put('/predictors/config', json={'features': []}).status_code)
        schedule = self.client.get('/retraining/schedule').json()
        self.assertEqual('Manual', schedule['cadencia'])
        self.assertEqual('No programada', schedule['proxima_ejecucion_programada'])


if __name__ == '__main__':
    unittest.main()
