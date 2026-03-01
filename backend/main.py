from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from pathlib import Path
from typing import Any, Literal

import torch
from celery.result import AsyncResult
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError

from model import runtime_info
from tasks import async_decompose, celery_app, get_task_dir, get_zip_path

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)

MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
TASK_STALE_SECONDS = int(os.getenv("TASK_STALE_SECONDS", "900"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(
    title="Image Layer Decomposition API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timeout_middleware(request: Request, call_next: Any) -> Response:
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
        return response
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={
                "detail": f"Request timed out after {REQUEST_TIMEOUT_SECONDS} seconds",
            },
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/api/health")
async def health() -> dict[str, Any]:
    info = runtime_info()
    gpu_info = _get_gpu_info()

    return {
        "status": "ok",
        "model": {
            "id": info.model_id,
            "device": info.device,
            "dtype": info.dtype,
            "loaded": info.loaded,
        },
        "gpu": gpu_info,
    }


@app.post("/api/decompose", response_model=None)
async def decompose(
    request: Request,
    image: UploadFile = File(...),
    num_layers: int = Form(default=4),
    inference_preset: str = Form(default="balanced"),
    device_mode: str = Form(default="auto"),
    resolution: int = Form(default=512),
    num_inference_steps: int = Form(default=24),
    true_cfg_scale: float = Form(default=3.0),
    use_en_prompt: bool = Form(default=True),
    cfg_normalize: bool = Form(default=True),
) -> Response:
    _validate_num_layers(num_layers=num_layers)
    _validate_upload_metadata(upload=image)
    inference_options = _build_inference_options(
        inference_preset=inference_preset,
        device_mode=device_mode,
        resolution=resolution,
        num_inference_steps=num_inference_steps,
        true_cfg_scale=true_cfg_scale,
        use_en_prompt=use_en_prompt,
        cfg_normalize=cfg_normalize,
    )

    file_bytes = await image.read(MAX_FILE_SIZE_BYTES + 1)
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB upload limit",
        )

    _validate_uploaded_image(file_bytes=file_bytes)

    try:
        task = async_decompose.delay(
            image_b64=base64.b64encode(file_bytes).decode("utf-8"),
            original_filename=image.filename or "upload.png",
            num_layers=num_layers,
            inference_options=inference_options,
        )
        _store_task_submission(task.id)
    except Exception as exc:
        LOGGER.exception("Failed to enqueue decomposition task: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Task queue unavailable. Verify Redis/Celery services. Error: {exc}",
        ) from exc
    task_id = task.id

    force_async = request.headers.get("X-Async-Only", "").strip().lower() in {"1", "true", "yes"}
    if force_async:
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task_id,
                "status": "processing",
                "status_url": str(request.url_for("task_status", task_id=task_id)),
                "download_url": str(request.url_for("download_result", task_id=task_id)),
            },
        )

    result = AsyncResult(task_id, app=celery_app)
    deadline = time.monotonic() + REQUEST_TIMEOUT_SECONDS

    while time.monotonic() < deadline:
        if result.successful():
            payload = _safe_result_payload(result.result)
            zip_path = Path(payload["zip_path"])
            if not zip_path.exists():
                raise HTTPException(status_code=500, detail="Task completed but ZIP file is missing")

            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename=payload.get("zip_name", "layers.zip"),
            )

        if result.failed():
            raise HTTPException(status_code=500, detail=_extract_task_error(result))

        await asyncio.sleep(1.0)

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task_id,
            "status": "processing",
            "status_url": str(request.url_for("task_status", task_id=task_id)),
            "download_url": str(request.url_for("download_result", task_id=task_id)),
        },
    )


@app.get("/api/status/{task_id}", name="task_status")
async def task_status(request: Request, task_id: str) -> dict[str, Any]:
    result = AsyncResult(task_id, app=celery_app)
    zip_path = get_zip_path(task_id)
    progress = _extract_progress(result)
    state_message = _extract_state_message(result)
    events = _extract_events(result)
    updated_age_seconds = _get_task_update_age_seconds(result)
    raw_state = str(result.state)

    if result.successful() or zip_path.exists():
        payload = _safe_result_payload(result.result) if result.successful() else {}
        layers = payload.get("layers", _discover_layer_files(task_id=task_id))
        layer_urls = _build_layer_urls(request=request, task_id=task_id, layers=layers)
        return {
            "task_id": task_id,
            "status": "done",
            "progress": 100,
            "download_url": str(request.url_for("download_result", task_id=task_id)),
            "layers": layers,
            "layer_urls": layer_urls,
            "events": payload.get("events", events),
            "state": raw_state,
        }

    if result.failed() or raw_state == "REVOKED":
        return {
            "task_id": task_id,
            "status": "error",
            "progress": progress,
            "message": state_message,
            "error": _extract_task_error(result),
            "events": events,
            "state": raw_state,
        }

    age_seconds = _get_task_submission_age_seconds(task_id)
    stale_age_seconds: int | None = None
    if raw_state == "PENDING":
        stale_age_seconds = age_seconds
    elif raw_state in {"STARTED", "PROGRESS", "RETRY"}:
        stale_age_seconds = updated_age_seconds if updated_age_seconds is not None else age_seconds

    if stale_age_seconds is not None and stale_age_seconds > TASK_STALE_SECONDS and raw_state in {
        "PENDING",
        "STARTED",
        "PROGRESS",
        "RETRY",
    }:
        return {
            "task_id": task_id,
            "status": "error",
            "progress": progress,
            "message": state_message or "Task appears stalled",
            "error": (
                f"Task appears stalled (no updates for {stale_age_seconds}s, state={raw_state}). "
                "Please retry with Safe preset or Force CPU Mode."
            ),
            "events": events,
            "state": raw_state,
        }

    mapped_status: Literal["pending", "processing"] = (
        "processing" if raw_state in {"STARTED", "RETRY", "PROGRESS"} else "pending"
    )
    return {
        "task_id": task_id,
        "status": mapped_status,
        "progress": progress,
        "message": state_message,
        "events": events,
        "state": raw_state,
    }


@app.get("/api/download/{task_id}", name="download_result")
async def download_result(task_id: str) -> FileResponse:
    zip_path = get_zip_path(task_id)
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Result ZIP is not available for this task")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=f"{task_id}_layers.zip",
    )


@app.get("/api/layer/{task_id}/{layer_name}", name="download_layer")
async def download_layer(task_id: str, layer_name: str) -> FileResponse:
    if Path(layer_name).name != layer_name or not layer_name.lower().endswith(".png"):
        raise HTTPException(status_code=400, detail="Invalid layer file name")

    layer_path = get_task_dir(task_id) / layer_name
    if not layer_path.exists():
        raise HTTPException(status_code=404, detail="Layer file not found")

    return FileResponse(
        path=layer_path,
        media_type="image/png",
        filename=layer_name,
    )


def _validate_num_layers(num_layers: int) -> None:
    if not 2 <= num_layers <= 8:
        raise HTTPException(status_code=422, detail="num_layers must be between 2 and 8")


def _validate_upload_metadata(upload: UploadFile) -> None:
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension. Allowed: JPG, PNG, WEBP",
        )

    if upload.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported MIME type. Allowed: image/jpeg, image/png, image/webp",
        )


def _validate_uploaded_image(file_bytes: bytes) -> None:
    try:
        from io import BytesIO

        with Image.open(BytesIO(file_bytes)) as candidate:
            candidate.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image") from exc


def _extract_task_error(result: AsyncResult) -> str:
    if isinstance(result.result, Exception):
        return str(result.result)
    if isinstance(result.result, dict) and "error" in result.result:
        return str(result.result["error"])
    return "Image decomposition task failed"


def _safe_result_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Task payload is malformed")
    if "zip_path" not in payload:
        raise HTTPException(status_code=500, detail="Task payload does not include ZIP path")
    return payload


def _extract_progress(result: AsyncResult) -> int:
    if result.successful():
        return 100
    meta = result.info if isinstance(result.info, dict) else {}
    raw_progress = meta.get("progress", 0)
    try:
        progress = int(raw_progress)
    except (TypeError, ValueError):
        progress = 0
    return max(0, min(progress, 100))


def _extract_state_message(result: AsyncResult) -> str | None:
    meta = result.info if isinstance(result.info, dict) else {}
    message = meta.get("message")
    return str(message) if message else None


def _extract_events(result: AsyncResult) -> list[str]:
    meta = result.info if isinstance(result.info, dict) else {}
    raw_events = meta.get("events", [])
    if not isinstance(raw_events, list):
        return []
    return [str(item) for item in raw_events][-20:]


def _get_task_update_age_seconds(result: AsyncResult) -> int | None:
    meta = result.info if isinstance(result.info, dict) else {}
    raw_updated_at = meta.get("updated_at")
    try:
        if raw_updated_at is None:
            return None
        updated_ts = int(raw_updated_at)
        return max(0, int(time.time()) - updated_ts)
    except (TypeError, ValueError):
        return None


def _discover_layer_files(task_id: str) -> list[str]:
    task_dir = get_task_dir(task_id)
    if not task_dir.exists():
        return []
    return sorted(
        file.name for file in task_dir.iterdir() if file.is_file() and file.suffix.lower() == ".png"
    )


def _build_layer_urls(request: Request, task_id: str, layers: list[str]) -> list[str]:
    return [str(request.url_for("download_layer", task_id=task_id, layer_name=layer)) for layer in layers]


def _build_inference_options(
    inference_preset: str,
    device_mode: str,
    resolution: int,
    num_inference_steps: int,
    true_cfg_scale: float,
    use_en_prompt: bool,
    cfg_normalize: bool,
) -> dict[str, Any]:
    preset = inference_preset.strip().lower()
    preset_map: dict[str, dict[str, Any]] = {
        "safe": {"resolution": 384, "num_inference_steps": 16, "true_cfg_scale": 2.5},
        "balanced": {"resolution": 512, "num_inference_steps": 24, "true_cfg_scale": 3.0},
        "quality": {"resolution": 640, "num_inference_steps": 32, "true_cfg_scale": 3.5},
        "custom": {},
    }
    if preset not in preset_map:
        raise HTTPException(status_code=422, detail="inference_preset must be one of: safe, balanced, quality, custom")

    mode = device_mode.strip().lower()
    if mode not in {"auto", "cpu", "cuda", "balanced"}:
        raise HTTPException(status_code=422, detail="device_mode must be one of: auto, cpu, cuda, balanced")

    options: dict[str, Any] = {
        "use_en_prompt": use_en_prompt,
        "cfg_normalize": cfg_normalize,
        "device_mode": mode,
    }
    options.update(preset_map[preset])
    if preset == "custom":
        options["resolution"] = resolution
        options["num_inference_steps"] = num_inference_steps
        options["true_cfg_scale"] = true_cfg_scale

    try:
        options["resolution"] = int(options.get("resolution", 512))
        options["num_inference_steps"] = int(options.get("num_inference_steps", 24))
        options["true_cfg_scale"] = float(options.get("true_cfg_scale", 3.0))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid inference tuning values") from exc

    return options


def _get_gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "vram_used_mb": 0,
            "vram_total_mb": 0,
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    total_bytes = int(props.total_memory)

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
        used_bytes = total_bytes - int(free_bytes)
    except RuntimeError:
        used_bytes = 0

    return {
        "available": True,
        "name": props.name,
        "vram_used_mb": round(used_bytes / (1024 * 1024), 2),
        "vram_total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def _store_task_submission(task_id: str) -> None:
    try:
        backend = celery_app.backend
        client = getattr(backend, "client", None)
        if client is None:
            return
        client.setex(f"task:submitted:{task_id}", 60 * 60 * 24, str(int(time.time())))
    except Exception as exc:
        LOGGER.warning("Could not store task submission timestamp for %s: %s", task_id, exc)


def _get_task_submission_age_seconds(task_id: str) -> int | None:
    try:
        backend = celery_app.backend
        client = getattr(backend, "client", None)
        if client is None:
            return None
        raw_value = client.get(f"task:submitted:{task_id}")
        if raw_value is None:
            return None
        created_ts = int(raw_value.decode("utf-8") if isinstance(raw_value, bytes) else raw_value)
        return max(0, int(time.time()) - created_ts)
    except Exception as exc:
        LOGGER.warning("Could not read task submission timestamp for %s: %s", task_id, exc)
        return None
