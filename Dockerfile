# syntax=docker/dockerfile:1.4

# =========================================================
# Stage 1: Builder Environment
# =========================================================
FROM python:3.10-slim AS builder

# Set cache directory paths for ccache and Nuitka
ENV CCACHE_DIR=/root/.cache/ccache \
    NUITKA_CACHE_DIR=/root/.cache/nuitka

# Install C/C++ build tools required by Nuitka and torch-two-sample
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    ccache \
    patchelf \
    git \
    python3-dev

WORKDIR /build

# 1. Install PyTorch, Nuitka, and core dependencies (using BuildKit pip cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    nuitka \
    torch \
    torchvision \
    scipy \
    tqdm \
    Pillow \
    "numpy<2.0"

# 2. Build and install the torch-two-sample C++ extension
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone https://github.com/josipd/torch-two-sample.git \
    && cd torch-two-sample \
    && python setup.py install \
    && cd .. && rm -rf torch-two-sample

# 3. Copy project files into the build environment
COPY . /build

# 4. Compile the project with Nuitka (persisting ccache and Nuitka compilation caches)
# Flags:
#   --standalone: Bundles the executable and shared libraries into a folder
#   --enable-plugin=torch / numpy: Ensures PyTorch and NumPy C-extensions/DLLs are included
#   --include-package: Ensures local submodules are included in the build
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/nuitka \
    python3 -m nuitka \
    --standalone \
    --enable-plugin=torch \
    --enable-plugin=numpy \
    --include-package=data \
    --include-package=models \
    --include-package=utils \
    --output-dir=/build/dist \
    --output-filename=shift_detector \
    experiment.py

# =========================================================
# Stage 2: Minimal Runtime Image (Ubuntu 24.04 for GLIBC 2.38+)
# =========================================================
FROM ubuntu:24.04 AS runner

# Install basic runtime C-libraries needed by Pillow/OpenCV/PyTorch
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates

WORKDIR /app

# Copy the compiled standalone distribution from the builder stage
COPY --from=builder /build/dist/experiment.dist /app/experiment.dist

# Set working directory inside the compiled distribution
WORKDIR /app/experiment.dist

# Define the default entrypoint to run your compiled binary
ENTRYPOINT ["./shift_detector"]