# Dynaverse GPU development image
# Base image pinned by digest (recorded in env/versions.lock).
# Rebuild from source on every new GPU target (sm_90 / sm_120); never copy artifacts.
ARG CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu22.04@sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97
FROM ${CUDA_BASE_IMAGE}

ARG RUST_VERSION=1.89.0
ARG CMAKE_VERSION=3.30.5
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV DEBIAN_FRONTEND=noninteractive

# Build tools only; no NVIDIA driver packages here (driver comes from the host).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
        ca-certificates \
        pkg-config \
        ninja-build \
        clang \
        clang-format \
        llvm-dev \
        libclang-dev \
        python3 \
        python3-pip \
        python3-venv \
        sudo \
        gdb \
        vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# CMake pinned binary release (Ubuntu jammy apt cmake is too old for many projects)
RUN wget -q "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" -O /tmp/cmake.tgz \
    && tar -xzf /tmp/cmake.tgz -C /usr/local --strip-components=1 \
    && rm /tmp/cmake.tgz && cmake --version | head -1

# Non-root user
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"${USERNAME}" \
    && chmod 440 /etc/sudoers.d/"${USERNAME}"

# Rust toolchain (pinned). RUSTUP_HOME/CARGO_HOME are passed explicitly to the
# installer because `su - $USER` drops the image-level ENV.
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:${PATH}
RUN chmod 777 /usr/local \
    && su "${USERNAME}" -c "export RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo; curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION} --profile minimal --no-modify-path" \
    && chown -R "${USER_UID}:${USER_GID}" /usr/local/rustup /usr/local/cargo \
    && /usr/local/cargo/bin/rustc --version && /usr/local/cargo/bin/cargo --version

USER ${USERNAME}
WORKDIR /workspace

# CUDA_ARCH is passed at build/run time via env/local.env (120) or env/h200.env (90).
ARG CUDA_ARCH=120
ENV CUDA_ARCH=${CUDA_ARCH}
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}

CMD ["bash"]
