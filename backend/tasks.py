from __future__ import annotations

import base64
import logging
import os
import shutil
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from celery import Celery
from celery.exceptions import SoftTimeLimitExceeded
from PIL import Image, UnidentifiedImageError

from model import decompose_image

LOGGER = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/1")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/output"))
OUTPUT_RETENTION_SECONDS = int(os.getenv("OUTPUT_RETENTION_SECONDS", "3600"))
INFERENCE_HEARTBEAT_SECONDS = int(os.getenv("INFERENCE_HEARTBEAT_SECONDS", "20"))

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
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=False,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT_SECONDS", "1800")),
    task_time_limit=int(os.getenv("TASK_HARD_TIME_LIMIT_SECONDS", "2100")),
    broker_transport_options={
        "visibility_timeout": int(os.getenv("BROKER_VISIBILITY_TIMEOUT_SECONDS", "3600")),
    },
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
def async_decompose(
    self: Any,
    image_b64: str,
    original_filename: str,
    num_layers: int,
    inference_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_id = str(self.request.id)
    task_dir = get_task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    options = inference_options or {}
    events: list[str] = []
    progress_lock = threading.Lock()

    def update_progress(progress: int, message: str) -> None:
        with progress_lock:
            events.append(message)
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "message": message,
                    "events": events[-20:],
                    "updated_at": int(time.time()),
                },
            )

    try:
        update_progress(10, "Decoding image")
        image = _decode_image(image_b64=image_b64)

        update_progress(
            25,
            "Preparing inference (model loading/inference may take several minutes, especially in CPU mode)",
        )
        heartbeat_stop = threading.Event()
        inference_started_at = time.monotonic()

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(INFERENCE_HEARTBEAT_SECONDS):
                elapsed_seconds = int(time.monotonic() - inference_started_at)
                heartbeat_progress = min(70, 25 + max(0, elapsed_seconds // 20))
                update_progress(
                    int(heartbeat_progress),
                    f"Inference running ({elapsed_seconds}s elapsed, first run may download model files)",
                )

        heartbeat_thread = threading.Thread(target=heartbeat_loop, name=f"inference-heartbeat-{task_id}")
        heartbeat_thread.daemon = True
        heartbeat_thread.start()
        try:
            layers = decompose_image(image=image, num_layers=num_layers, options=options)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)

        update_progress(80, "Saving layer files")
        layer_paths = _save_layers(task_dir=task_dir, layers=layers)
        zip_path = _create_zip(task_dir=task_dir, layer_paths=layer_paths)

        update_progress(95, "Finalizing output")
        schedule_cleanup(task_dir=task_dir)
    except SoftTimeLimitExceeded as exc:
        cleanup_task_dir(task_dir)
        LOGGER.exception("Task %s timed out: %s", task_id, exc)
        raise RuntimeError(
            "Task exceeded execution time limit. Reduce resolution/steps or enable Force CPU Mode."
        ) from exc
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
        "events": events[-20:],
    }
