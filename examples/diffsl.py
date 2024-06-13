import diffsol

builder = diffsol.OdeBuilder().rtol(1e-6).p([0.1])
test = builder.wip_diffsl_solve("""
    in = [a]
    a { 1 }
    u { 1.0 }
    F { -a*u }
    out { u }
""")

print(test)