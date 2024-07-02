FROM rust:1.79-bookworm

RUN apt-get update
RUN apt install -y --no-install-recommends \
    cmake \
    llvm-14-dev \
    libpolly-14-dev \
    libclang-14-dev \
    python3.11-venv \
    python3.11-dev \
    && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /venv
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH
RUN pip install maturin
