#!/usr/bin/env bash
# ── GPU VM Setup — Docker Engine + NVIDIA Container Toolkit ──────────────────
#
# Run once on a fresh Ubuntu 22.04 / 24.04 GPU VM before using docker compose.
#
# What this script does:
#   1. Installs Docker Engine from Docker's official APT repo
#   2. Adds the NVIDIA Container Toolkit APT repo and installs it
#   3. Configures Docker daemon to use the NVIDIA runtime
#   4. Restarts Docker and verifies GPU passthrough
#
# IMPORTANT — read before running:
#   • Docker must be installed via APT, NOT via Ubuntu Snap.
#     The Snap sandbox blocks access to /dev/nvidia* — GPU passthrough silently
#     fails.  If Docker is already installed via Snap, purge it first:
#       sudo snap remove --purge docker
#   • The VM must have NVIDIA drivers installed on the host OS (not this script).
#     Cloud VMs (GCP, AWS, Azure) with GPU usually come with drivers pre-installed.
#     Verify with: nvidia-smi
#   • '--gpus all' is the correct modern syntax.  '--runtime=nvidia' is the
#     deprecated nvidia-docker2 form and should not be used in new deployments.
#   • 'capabilities: [gpu]' in docker-compose is mandatory — without it the GPU
#     is silently ignored and no error is raised.
#
# After this script completes, run the training job with:
#   docker compose --profile train run --rm --build train
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

echo "==> [1/5] Installing Docker Engine from Docker's official APT repo..."

# Remove any snap-installed Docker that would block GPU passthrough
if snap list docker &>/dev/null 2>&1; then
    echo "  Detected Docker installed via Snap — removing it first..."
    sudo snap remove --purge docker
fi

sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key and APT repo
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -qq
sudo apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "  Docker $(docker --version) installed."

echo ""
echo "==> [2/5] Adding NVIDIA Container Toolkit APT repo..."

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update -qq

echo ""
echo "==> [3/5] Installing NVIDIA Container Toolkit..."

sudo apt-get install -y nvidia-container-toolkit

echo ""
echo "==> [4/5] Configuring Docker daemon to use NVIDIA runtime and restarting..."

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "==> [5/5] Smoke test — running nvidia-smi inside a container..."

# Add current user to docker group so future commands don't need sudo
sudo usermod -aG docker "$USER"

sudo docker run --rm --gpus all \
    nvidia/cuda:12.4.1-base-ubuntu22.04 \
    nvidia-smi

echo ""
echo "==> Setup complete. GPU passthrough is working."
echo ""
echo "    Run training:"
echo "      docker compose --profile train run --rm --build train"
echo ""
echo "    Verify XGBoost detects CUDA inside the training image:"
echo "      docker build -f AIModel/Dockerfile.training -t spike-train AIModel/"
echo "      docker run --rm --gpus all -e DEVICE=cuda spike-train \\"
echo "        python3 -c \"from spike_classifier import _resolve_device; print(_resolve_device('cuda'))\""
