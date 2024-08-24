FROM nvidia/cuda:11.7.1-base-ubuntu22.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# Copy requirements.txt first to leverage Docker caching for dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Clone SAM 2 repository and install it
RUN git clone https://github.com/facebookresearch/segment-anything-2.git \
    && cd segment-anything-2 && pip install -e .

# Download SAM 2 checkpoint
RUN mkdir -p checkpoints \
    && wget -O checkpoints/sam2_hiera_small.pt https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

# Copy the rest of the application code
COPY . .

# Run your application
CMD ["python3", "main.py"]
