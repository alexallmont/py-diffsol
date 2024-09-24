from diffsol import nalgebra_dense_lu_f64 as ds

context = ds.Context(
"""
    in = [a]
    a { 1 }
    u { 1.0 }
    F { -a*u }
    out { u }
"""
)

builder = ds.Builder().rtol(1e-6).p([0.1]).h0(5.0)
problem = builder.build_diffsl(context)
solver = diffsol.Bdf()
result = solver.solve(problem)
print(result.t, result.y)
