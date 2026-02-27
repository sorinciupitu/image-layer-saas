# Image Layer Decomposition SaaS

Production-ready full-stack app for decomposing a single image into multiple transparent PNG layers (RGBA), powered by `Qwen/Qwen-Image-Layered` with local inference (no ComfyUI).

Stack:
- Backend: FastAPI + Celery + Redis + Hugging Face Diffusers
- Frontend: Vanilla JS/HTML/CSS (no build step)
- Runtime: Docker Compose + NVIDIA GPU
- Deployment: Windows Server/Windows 11 host + WSL2 Ubuntu + GitHub Actions self-hosted runner

## Features

- Upload JPG/PNG/WEBP images (up to 20MB)
- Choose layer count (`2-8`)
- Async decomposition with status polling and progress percentage
- Per-layer PNG download and full ZIP download
- Transparent checkerboard previews in UI
- GPU-aware health endpoint with VRAM stats

## Repository Layout

- `backend/main.py`: FastAPI endpoints, validation, status/download APIs
- `backend/model.py`: model singleton, inference, CUDA/CPU fallback
- `backend/tasks.py`: Celery async decomposition and output retention
- `frontend/index.html`: single-page interface
- `frontend/app.js`: upload/polling/results UI logic
- `frontend/style.css`: responsive dark UI
- `frontend/nginx.conf`: static hosting + `/api` reverse proxy
- `server-setup/setup-wsl.sh`: one-shot WSL2 bootstrap
- `server-setup/fix-nvidia-docker.sh`: NVIDIA runtime fix/validation
- `server-setup/.wslconfig`: Windows-side WSL resource tuning
- `.github/workflows/deploy.yml`: auto-deploy workflow for self-hosted runner

## Prerequisites

Host (Windows 11/Server):
- Docker Desktop installed with WSL2 backend enabled
- Latest NVIDIA drivers (WSL2-compatible)
- WSL2 Ubuntu distribution installed

Inside WSL2 Ubuntu:
- Docker Engine / Docker Compose plugin
- `nvidia-container-toolkit`
- Git, curl, wget, build-essential

Important performance rule:
- Keep the project and Docker volumes inside WSL2 filesystem, for example `/home/<username>/image-layer-saas`.
- Do not run under `/mnt/c/...` due to cross-filesystem I/O penalties.

## First-Time Server Setup

Run in WSL2 Ubuntu:

```bash
chmod +x server-setup/*.sh
./server-setup/setup-wsl.sh
```

If Docker and toolkit are already installed, run only:

```bash
./server-setup/fix-nvidia-docker.sh
```

Also place:
- `server-setup/.wslconfig` at `C:\Users\<username>\.wslconfig` on Windows host (not inside WSL).
- Restart WSL after any `.wslconfig` change:

```powershell
wsl --shutdown
wsl
```

## Installation and Run

1. Clone into WSL2 home:

```bash
cd ~
git clone <your-repo-url> image-layer-saas
cd image-layer-saas
```

2. Configure environment:

```bash
cp .env.example .env
```

3. Edit `.env` for your environment (paths, model, runtime tuning if needed).

4. Start services:

```bash
docker compose up --build -d
```

5. Open:
- Frontend: `http://localhost`
- Backend health: `http://localhost:8000/api/health`

## Environment Variables

From `.env.example`:
- `APP_ENV`, `LOG_LEVEL`
- `REQUEST_TIMEOUT_SECONDS`, `MAX_UPLOAD_MB`
- `HF_MODEL_ID`, `HF_HOME`, `HF_TOKEN`
- `REDIS_URL`, `CELERY_RESULT_BACKEND`
- `OUTPUT_DIR`, `OUTPUT_RETENTION_SECONDS`
- `PRELOAD_MODEL`

## API Documentation

Base URL:
- Local backend: `http://localhost:8000`
- If calling through frontend Nginx: `http://localhost`

### `POST /api/decompose`

Multipart form-data:
- `image` (required): `.jpg`, `.jpeg`, `.png`, `.webp` (max 20MB)
- `num_layers` (optional): integer `2-8`, default `4`

Behavior:
- Default: server can return ZIP immediately (`200`) if task finishes within timeout.
- Async mode: send header `X-Async-Only: true` to always get `202` with task info.

Example (async):

```bash
curl -X POST "http://localhost:8000/api/decompose" \
  -H "X-Async-Only: true" \
  -F "image=@/path/to/image.png" \
  -F "num_layers=4"
```

Example response (`202`):

```json
{
  "task_id": "7f8e...",
  "status": "processing",
  "status_url": "http://localhost:8000/api/status/7f8e...",
  "download_url": "http://localhost:8000/api/download/7f8e..."
}
```

### `GET /api/status/{task_id}`

Returns:
- `pending`
- `processing` (with `progress` and optional message)
- `done` (includes `download_url`, `layers`, `layer_urls`)
- `error` (includes error message)

Example:

```bash
curl "http://localhost:8000/api/status/<task_id>"
```

### `GET /api/download/{task_id}`

Downloads ZIP for completed task:

```bash
curl -L "http://localhost:8000/api/download/<task_id>" -o layers.zip
```

### `GET /api/layer/{task_id}/{layer_name}`

Downloads a single PNG layer:

```bash
curl -L "http://localhost:8000/api/layer/<task_id>/layer_01.png" -o layer_01.png
```

### `GET /api/health`

Returns API/model/GPU status:

```bash
curl "http://localhost:8000/api/health"
```

Sample response:

```json
{
  "status": "ok",
  "model": {
    "id": "Qwen/Qwen-Image-Layered",
    "device": "cuda",
    "dtype": "float16",
    "loaded": true
  },
  "gpu": {
    "available": true,
    "name": "NVIDIA GeForce RTX 3060",
    "vram_used_mb": 1234.0,
    "vram_total_mb": 12288.0
  }
}
```

## GitHub Actions Auto-Deploy Setup

Workflow file:
- `.github/workflows/deploy.yml`

Trigger:
- Every push to `main`

Runner:
- `self-hosted` runner on your Windows server

Required repository secrets:
- `DEPLOY_PATH`: absolute WSL path to project (example: `/home/username/image-layer-saas`)
- `ENV_FILE_PATH`: absolute path to real `.env` on server
- `HEALTH_CHECK_URL`: example `http://localhost:8000/api/health`

Deployment flow in workflow:
1. Pull latest code from `main`
2. Copy real `.env` into repo root
3. `docker compose down`
4. `docker compose up --build -d`
5. Poll health endpoint every 5s up to 3 minutes

Runner registration overview:
1. In GitHub repo: `Settings -> Actions -> Runners -> New self-hosted runner`
2. Choose OS/arch and follow setup script
3. Install and run runner from WSL2 shell if you want Linux/WSL execution context
4. Confirm runner shows `Idle/Online` in GitHub UI

## Troubleshooting

### 1. VRAM / Out of memory

Symptoms:
- Worker crashes during inference
- CUDA OOM errors in logs

Actions:
- Reduce concurrent load (`celery` concurrency is already `1`)
- Ensure no other GPU-heavy workloads are running
- Keep FP16 mode enabled
- Restart services:

```bash
docker compose restart backend worker
```

### 2. Model download errors

Symptoms:
- Timeout / 401 / model load failures

Actions:
- Verify internet connectivity in WSL2
- Set `HF_TOKEN` in `.env` if required
- Clear corrupted cache and rebuild:

```bash
docker compose down
docker volume rm image-layer-saas_hf_cache || true
docker compose up --build -d
```

### 3. CUDA not found in containers

Actions:
- Verify toolkit and runtime:

```bash
./server-setup/fix-nvidia-docker.sh
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

- Ensure Docker Desktop is using WSL2 backend
- Ensure host NVIDIA driver supports WSL2

### 4. WSL2 GPU passthrough issues

Actions:
- In Windows PowerShell:

```powershell
wsl --shutdown
```

- Re-open WSL and retry `nvidia-smi`
- Update Windows GPU driver and reboot host if needed

### 5. RAM limit / WSL memory pressure

Actions:
- Use `server-setup/.wslconfig` on Windows host (`C:\Users\<username>\.wslconfig`)
- Restart WSL:

```powershell
wsl --shutdown
wsl
```

### 6. Deployment runner issues

Actions:
- Check runner service status
- Confirm repo secrets exist and are correct
- Confirm `DEPLOY_PATH` points to WSL project path and that `.env` file exists at `ENV_FILE_PATH`
- Inspect Actions logs for failing step

## Notes

- This implementation uses local inference through Hugging Face Diffusers and does not require ComfyUI.
- Output files are retained under `/tmp/output/<task_id>/` and cleaned after retention window.
