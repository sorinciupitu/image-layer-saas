#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="/etc/nvidia-container-runtime/config.toml"
TEST_IMAGE="nvidia/cuda:12.1.0-base-ubuntu22.04"
TEST_CMD="nvidia-smi"

echo "[1/4] Validating environment..."
if ! grep -qi microsoft /proc/version; then
  echo "Error: This script must be run inside WSL2 Ubuntu (not PowerShell)."
  exit 1
fi

if ! grep -qi ubuntu /etc/os-release; then
  echo "Error: This script is intended for Ubuntu inside WSL2."
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found. Install Docker first."
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: $CONFIG_FILE not found. Install nvidia-container-toolkit first."
  exit 1
fi

echo "[2/4] Updating no-cgroups setting..."
current_setting="$(grep -E '^\s*no-cgroups\s*=' "$CONFIG_FILE" || true)"

if echo "$current_setting" | grep -q "false"; then
  echo "no-cgroups is already set to false."
else
  sudo sed -i -E 's/^\s*no-cgroups\s*=\s*true/no-cgroups = false/g' "$CONFIG_FILE"
  if ! grep -Eq '^\s*no-cgroups\s*=' "$CONFIG_FILE"; then
    echo "no-cgroups = false" | sudo tee -a "$CONFIG_FILE" >/dev/null
  fi
  if ! grep -Eq '^\s*no-cgroups\s*=\s*false' "$CONFIG_FILE"; then
    echo "Error: failed to set no-cgroups = false in $CONFIG_FILE"
    exit 1
  fi
  echo "Updated $CONFIG_FILE successfully."
fi

echo "[3/4] Restarting Docker daemon if available..."
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^docker.service'; then
  sudo systemctl restart docker || true
fi

echo "[4/4] Verifying GPU access from Docker..."
if docker run --rm --gpus all "$TEST_IMAGE" "$TEST_CMD"; then
  echo "Success: NVIDIA runtime is working in Docker containers."
else
  echo "Failure: GPU test failed."
  echo "Next steps:"
  echo "1. Confirm Windows NVIDIA driver is up to date."
  echo "2. Ensure Docker Desktop uses the WSL2 backend."
  echo "3. Re-run this script, then retry:"
  echo "   docker run --rm --gpus all $TEST_IMAGE $TEST_CMD"
  exit 1
fi
