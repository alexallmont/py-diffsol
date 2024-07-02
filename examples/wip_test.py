from copy import copy
import diffsol

context = diffsol.OdeSolverContext("""
    in = [a]
    a { 1 }
    u { 1.0 }
    F { -a*u }
    out { u }
""")

builder = diffsol.OdeBuilder().rtol(1e-6).p([0.1])
problem = builder.build_diffsl(context)
