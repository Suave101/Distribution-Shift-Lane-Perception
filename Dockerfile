# syntax=docker/dockerfile:1.4

# =========================================================
# Stage 1: Builder Environment
# =========================================================
FROM python:3.10-slim AS builder

# Set cache directory paths for ccache and Nuitka
ENV CCACHE_DIR=/root/.cache/ccache \
    NUITKA_CACHE_DIR=/root/.cache/nuitka

# Install C/C++ build tools required by Nuitka and torch-two-sample
# Force IPv4 to prevent WSL2 TCP network timeouts during apt updates
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::ForceIPv4=true update && apt-get install -y --no-install-recommends \
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

# 4. Compile experiment.py with Nuitka (persisting ccache and Nuitka compilation caches)
RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/nuitka \
    python3 -m nuitka \
    --standalone \
    --remove-output \
    --lto=no \
    --jobs=auto \
    --enable-plugin=torch \
    --enable-plugin=numpy \
    --nofollow-import-to=torch \
    --nofollow-import-to=torchvision \
    --nofollow-import-to=scipy \
    --include-package=torch \
    --include-package=torchvision \
    --include-package=scipy \
    --include-package=data \
    --include-package=models \
    --include-package=utils \
    --output-dir=/build/dist \
    --output-filename=shift_detector \
    experiment.py

# 5. Compress the output directory into a single tarball
RUN tar -czf /build/experiment.tar.gz -C /build/dist experiment.dist

# =========================================================
# Stage 2: Minimal Runtime Image (Ubuntu 24.04 for GLIBC 2.38+)
# =========================================================
FROM ubuntu:24.04 AS runner

# Install basic runtime C-libraries needed by Pillow/OpenCV/PyTorch
# Force IPv4 to prevent WSL2 TCP network timeouts during apt updates
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get -o Acquire::ForceIPv4=true update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates

WORKDIR /app

# Copy the SINGLE compressed archive instead of tens of thousands of loose files
COPY --from=builder /build/experiment.tar.gz /app/

# Extract the archive in seconds and remove the tar file
RUN tar -xzf experiment.tar.gz && rm experiment.tar.gz

# Set working directory inside the compiled distribution
WORKDIR /app/experiment.dist

# Define the default entrypoint to run your compiled binary
ENTRYPOINT ["./shift_detector"]