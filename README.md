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

## TODO

- faer::Mat and matrix::sparse_faer::SparseColMat support
  - generic impl along with nalgebra::DMatrix
- Add GitHub actions
- Remove pub access?
- Review multithreaded access. Is sync/send safe?
