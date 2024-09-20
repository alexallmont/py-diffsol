# py-diffsol

Python wrapper for diffsol ODE solver library

## Usage

This MVP implements enough to build an ODE using DiffSl and solve using BDF.

Until the officially released, the best way to run is to clone this repo and
open the folder in VSCode. Follow the prompts to open in a dev container which
will set up build dependencies from the Dockerfile and VSCode extensions for
local development.

`launch.json` has a configuration for debugging the Rust code instantiated from
the Python example below, i.e. you can put breakpoints on any of the Rust
currently under development.

## Example

    $ maturin develop
    $ python
    >>> import diffsol
    >>> context = diffsol.OdeSolverContext(
    ... """
    ...     in = [a]
    ...     a { 1 }
    ...     u { 1.0 }
    ...     F { -a*u }
    ...     out { u }
    ... """
    ... )
    >>> builder = diffsol.OdeBuilder().rtol(1e-6).p([0.1]).h0(5.0)
    >>> problem = builder.build_diffsl(context)
    >>> solver = diffsol.Bdf()
    >>> result = solver.solve(problem)
    >>> print(result.t, result.y)

## Building and testing outside devcontainer

To build against a specific LLVM version, set the `LLVM_SYS_140_PREFIX` and `LLVM_DIR` environment variables.

To run `cargo test`, set `PYTHONPATH` to the location of your venv site packages to distinguish from the system environment.

For example, a native macOS build with a local `venv` could be configured with a `.cargo/config.toml` file containing:

    LLVM_SYS_140_PREFIX = "/opt/homebrew/opt/llvm@14"
    LLVM_DIR = "/opt/homebrew/opt/llvm@14"
    PYTHONPATH = "venv/lib/python3.12/site-packages"
