#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$HOME/image-layer-saas"

if ! grep -qi microsoft /proc/version; then
  echo "Error: setup-wsl.sh must be run inside WSL2 Ubuntu (not PowerShell)."
  exit 1
fi

if ! grep -qi ubuntu /etc/os-release; then
  echo "Error: setup-wsl.sh targets Ubuntu inside WSL2."
  exit 1
fi

echo "[1/9] Updating apt package index..."
sudo apt-get update
sudo apt-get upgrade -y

echo "[2/9] Installing base tooling..."
sudo apt-get install -y \
  apt-transport-https \
  ca-certificates \
  curl \
  wget \
  gnupg \
  lsb-release \
  git \
  build-essential

echo "[3/9] Installing Docker Engine from official Docker apt repository..."
if [ ! -f /usr/share/keyrings/docker-archive-keyring.gpg ]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
fi

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "[4/9] Adding user '$USER' to docker group..."
sudo usermod -aG docker "$USER"

echo "[5/9] Installing NVIDIA Container Toolkit..."
if [ ! -f /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg ]; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
fi

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

if command -v nvidia-ctk >/dev/null 2>&1; then
  sudo nvidia-ctk runtime configure --runtime=docker
fi

if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | grep -q '^docker.service'; then
  sudo systemctl restart docker || true
fi

echo "[6/9] Running NVIDIA Docker runtime fix..."
"$SCRIPT_DIR/fix-nvidia-docker.sh"

echo "[7/9] Installing GitHub Actions Runner dependencies..."
sudo apt-get install -y \
  libicu-dev \
  libkrb5-3 \
  zlib1g \
  libssl-dev

echo "[8/9] Creating project directory..."
mkdir -p "$PROJECT_DIR"

echo "[9/9] Setup complete."
echo
echo "Manual checklist:"
echo "1. Place .wslconfig in C:\\Users\\<username>\\ on the Windows host."
echo "2. Run 'wsl --shutdown' and then restart WSL2."
echo "3. Clone GitHub repo into ~/image-layer-saas."
echo "4. Create .env file from .env.example."
echo "5. Register GitHub Actions Self-hosted Runner."
echo "6. Run: docker-compose up --build -d"
