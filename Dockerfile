FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set working directory
WORKDIR /workspace

# Install basic tools
RUN apt-get update && apt-get install -y \
    git wget curl vim build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

COPY requirements.txt .

# Install python deps
RUN pip install -r requirements.txt

# Default command
CMD ["/bin/bash"]
