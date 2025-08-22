# -----------------------------
# Dockerfile for AIC2025 project
# Base: NVIDIA PyTorch with CUDA + cuDNN (includes glibc >= 2.35)
# -----------------------------
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Install basic tools
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements if you already have one
# (replace with your actual requirements.txt from aic2025)
COPY requirements.txt .

# Install python deps
RUN pip install -r requirements.txt

# Optional: install flash-attn from source (works well inside this image)
# RUN pip install flash-attn --no-build-isolation

# Default command
CMD ["/bin/bash"]
