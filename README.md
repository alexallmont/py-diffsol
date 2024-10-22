# py-diffsol

Python wrapper for [martinjrobins/diffsol](https://github.com/martinjrobins/diffsol)
ODE solver library.

## Usage

Note: this is in a pre-release state and wheels are not built yet. Leading up to
the initial release, the best way to run is to clone this repo and open the
folder in VSCode. Follow the prompts to open in a dev container which will set
up build dependencies from the Dockerfile and VSCode extensions for local
development.

The project uses [PyO3 maturin](https://github.com/PyO3/maturin) so a build and
tests can be run with:

    $ maturin develop --extras test
    $ pytest

Additionally, `launch.json` contains a debug configuration for debugging Rust
code in VSCode whilst a Python script is running. To use this put breakpoints on
the code Rust you want to interrogate and launch `Run Python Example`.

## Python API

The Rust API supports many matrix and solver types which are determined at
compile time when working in Rust. However, in order to build a Python wheel the
types need to be known up front. Currently py-diffsol has pre-baked the
following matrix-solver pairings:

- `nalgebra_dense_lu_f64`
- `faer_sparse_lu_f64`
- `faer_sparse_klu_f64`

Each name has four parts: 1) the matrix library; 2) whether the matrix is dense
or sparse; 3) the solver (currently LU or KLU); 4) the underlying scalar type.
The name is the submodule in `diffsol` that contains the various solver methods
and it is often convenient to import these with an alias, for example:

    from diffsol import faer_sparse_lu_f64 as ds

## Example

    $ maturin develop
    $ python
    >>> from diffsol import nalgebra_dense_lu_f64 as ds
    >>> context = ds.Context(
    ... """
    ...     in = [a]
    ...     a { 1 }
    ...     u { 1.0 }
    ...     F { -a*u }
    ...     out { u }
    ... """
    ... )
    >>> builder = ds.Builder().rtol(1e-6).p([0.1]).h0(5.0)
    >>> problem = builder.build_diffsl(context)
    >>> solver = ds.Bdf()
    >>> result = solver.solve(problem)
    >>> print(result.t, result.y)

## Building and testing outside devcontainer

To build against a specific LLVM version, set the `LLVM_SYS_160_PREFIX` and `LLVM_DIR` environment variables.

To run `cargo test`, set `PYTHONPATH` to the location of your venv site packages to distinguish from the system environment.

For example, a native macOS build with a local `venv` could be configured with a `.cargo/config.toml` file containing:

    LLVM_SYS_160_PREFIX = "/opt/homebrew/opt/llvm@16"
    LLVM_DIR = "/opt/homebrew/opt/llvm@16"
    PYTHONPATH = "venv/lib/python3.12/site-packages"
