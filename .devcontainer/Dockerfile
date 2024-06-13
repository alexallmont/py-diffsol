FROM python:3.12-bookworm

# Dev packages
RUN apt-get update && \
    apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Get Rust
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc

    #--no-install-recommends 
# Dev tools 
RUN apt-get update && \
    apt-get install -y cmake git && \
    rm -rf /var/lib/apt/lists/*

# LLVM for building Enzyme
RUN apt-get update && \
    apt-get install -y llvm libpolly-14-dev libclang-14-dev && \
    rm -rf /var/lib/apt/lists/*

# Install maturin
RUN python -m venv /venv
ENV VIRTUAL_ENV /venv
ENV PATH /venv/bin:$PATH 
RUN pip install maturin
