import diffsol

context = diffsol.OdeSolverContext(
"""
    in = [a]
    a { 1 }
    u { 1.0 }
    F { -a*u }
    out { u }
"""
)

builder = diffsol.OdeBuilder().rtol(1e-6).p([0.1]).h0(5.0)
problem = builder.build_diffsl(context)

solver = diffsol.Bdf()
result = solver.solve(problem)

print(result)