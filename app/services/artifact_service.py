"""Versioned artifacts shared by CLI trainers and the inference API."""
import json
import os
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from app import config


class TrainingInProgressError(RuntimeError):
    pass


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def active_run() -> dict:
    return read_json(config.ACTIVE_RUN_PATH)


def active_model_path(task: str, manifest: dict | None = None) -> Path:
    manifest = active_run() if manifest is None else manifest
    entry = manifest.get("models", {}).get(task)
    if entry:
        return config.TRAINED_MODELS_DIR / entry["model_path"]
    return config.GLOBAL_MODEL_PATH if task == "global" else config.COURSE_MODEL_PATH


def model_report(task: str) -> dict:
    entry = active_run().get("models", {}).get(task)
    if not entry:
        raise FileNotFoundError("No hay evaluación registrada. Ejecuta el reentrenamiento.")
    return read_json(config.TRAINED_MODELS_DIR / entry["report_path"])


@contextmanager
def training_lock():
    """OS lock released on process exit; coordinates CLI and API workers."""
    config.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with (config.TRAINED_MODELS_DIR / ".training.lock").open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise TrainingInProgressError("Ya hay un entrenamiento en ejecución.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
