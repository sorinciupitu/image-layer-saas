from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from celery import Celery
from PIL import Image, UnidentifiedImageError

from model import decompose_image

LOGGER = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/output"))
OUTPUT_RETENTION_SECONDS = int(os.getenv("OUTPUT_RETENTION_SECONDS", "3600"))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

celery_app = Celery(
    "image_layer_tasks",
    broker=REDIS_URL,
    backend=CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=OUTPUT_RETENTION_SECONDS,
    timezone="UTC",
    enable_utc=True,
)


def get_task_dir(task_id: str) -> Path:
    return OUTPUT_DIR / task_id


def get_zip_path(task_id: str) -> Path:
    return get_task_dir(task_id) / "layers.zip"


def cleanup_task_dir(task_dir: Path) -> None:
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


def schedule_cleanup(task_dir: Path, delay_seconds: int = OUTPUT_RETENTION_SECONDS) -> None:
    timer = threading.Timer(delay_seconds, cleanup_task_dir, args=(task_dir,))
    timer.daemon = True
    timer.start()


def _decode_image(image_b64: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64-encoded image payload") from exc

    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Could not decode uploaded image") from exc

    return image.convert("RGBA")


def _save_layers(task_dir: Path, layers: list[Image.Image]) -> list[Path]:
    layer_paths: list[Path] = []
    for idx, layer in enumerate(layers, start=1):
        layer_path = task_dir / f"layer_{idx:02d}.png"
        layer.save(layer_path, format="PNG")
        layer_paths.append(layer_path)
    return layer_paths


def _create_zip(task_dir: Path, layer_paths: list[Path]) -> Path:
    zip_path = task_dir / "layers.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for layer_path in layer_paths:
            zip_file.write(layer_path, arcname=layer_path.name)
    return zip_path


@celery_app.task(bind=True, name="async_decompose")
def async_decompose(self: Any, image_b64: str, original_filename: str, num_layers: int) -> dict[str, Any]:
    task_id = str(self.request.id)
    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        self.update_state(state="PROGRESS", meta={"progress": 10, "message": "Decoding image"})
        image = _decode_image(image_b64=image_b64)

        self.update_state(state="PROGRESS", meta={"progress": 45, "message": "Running model inference"})
        layers = decompose_image(image=image, num_layers=num_layers)

        self.update_state(state="PROGRESS", meta={"progress": 75, "message": "Saving layer files"})
        layer_paths = _save_layers(task_dir=task_dir, layers=layers)
        zip_path = _create_zip(task_dir=task_dir, layer_paths=layer_paths)

        self.update_state(state="PROGRESS", meta={"progress": 95, "message": "Finalizing output"})
        schedule_cleanup(task_dir=task_dir)
    except Exception as exc:
        cleanup_task_dir(task_dir)
        LOGGER.exception("Task %s failed: %s", task_id, exc)
        raise RuntimeError(f"Task failed: {exc}") from exc

    download_name = f"{Path(original_filename).stem or 'image'}_layers.zip"
    return {
        "task_id": task_id,
        "status": "done",
        "zip_path": str(zip_path),
        "zip_name": download_name,
        "layers": [layer_path.name for layer_path in layer_paths],
    }
