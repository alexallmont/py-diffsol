import diffsol
import numpy as np

context = diffsol.OdeSolverContext(
"""
    in = [r, k, y0]
    r { 1 } k { 1 } y0 { 1 }
    u { y0 }
    F { r*u*(1 - u/k) }
    out { u }
"""
)

r = 1.0
k = 10.0
y0 = 0.1
builder = diffsol.OdeBuilder().rtol(1e-6).p([r, k, y0])
problem = builder.build_diffsl(context)
solver = diffsol.Bdf()
result = solver.solve(problem)
for t, y in zip(result.t, result.y):
    expect = k*y0/(y0 + (k - y0)*np.exp(-r*t))
    err = np.abs(y[0] - expect)
    assert err < 1e-6
