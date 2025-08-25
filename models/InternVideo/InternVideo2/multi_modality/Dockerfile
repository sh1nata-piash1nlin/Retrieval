FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libegl1-mesa \
    libxrandr2 \
    libxrender1 \
    libx11-6 \
    libxext6 \
    libglib2.0-0 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create and activate virtual environment
RUN python3 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python-headless flash-attn

# Command to run
CMD ["bash"]