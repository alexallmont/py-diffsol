# py-diffsol

Python wrapper for diffsol ODE solver library

## TODO

 - For review on Friday:
   - DONE: pyoil wrapping builder, context, problem
   - REVIEW: ideal python API?
 - TODO
   - Use released version of PyOil3
   - Add GitHub actions
   - Remove pub access?
   - Review multithreaded access. Is sync/send safe?

## Example usage

    $ maturin develop
    $ python
    >>> import diffsol
    >>> builder = diffsol.OdeBuilder().rtol(1e-6).p([0.1])
    >>> test = builder.wip_diffsl_solve("""
    ... in = [a]
    ... a { 1 }
    ... u { 1.0 }
    ... F { -a*u }
    ... out { u }
    ... """)
    >>> print(test)
