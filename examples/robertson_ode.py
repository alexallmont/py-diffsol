import diffsol
import numpy as np

##    dy1/dt = -.04*y1 + 1.e4*y2*y3
##    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
##    dy3/dt = 3.e7*(y2)^2
## 3-species kinetics problem

context = diffsol.OdeSolverContext(
"""
    in = [a, b, c]
    a { 1 } b { 1 } c { 1 }
    u_i { y1 = 1.0, y2 = 0.0, y3 = 0.0 }
    F_i {
        -a*y1 + b*y2*y3,
        a*y1 - b*y2*y3 - c*y2*y2,
        c*y2*y2
    }
    out_i { u_i }
"""
)

builder = diffsol.OdeBuilder().rtol(1e-4).atol([1.e-8, 1.e-14, 1.e-6]).p([0.04, 1.e4, 3.e7])
problem = builder.build_diffsl(context)
solver = diffsol.Bdf()
t_eval = [0.0, 0.4, 4.0, 40.0, 400.0, 4000.0, 40000.0, 400000.0, 4000000.0, 2.0790e7, 4.0e7, 4.0e8, 4.0e9, 4.0e10]
ys = solver.solve_dense(problem, t_eval)

# these tolerances might not be correct since I didn't write out the expected values
# to this accuracy
rtol = 1e-4
atol = 1e-8
expected = [
    np.array([0.9899653, 3.470564e-05, 0.01]),
    np.array([0.9851641, 3.386242e-05, 0.01480205]),
    np.array([0.9055097, 2.240338e-05, 0.09446793]),
    np.array([0.7158017, 9.185037e-06, 0.2841892]),
    np.array([0.4505360, 3.223271e-06, 0.5494608]),
    np.array([0.1832299, 8.944378e-07, 0.8167692]),
    np.array([0.03898902, 1.622006e-07, 0.9610108]),
    np.array([0.004936383, 1.984224e-08, 0.9950636]),
    np.array([0.0005168093, 2.068293e-09, 0.9994832]),
    np.array([1.000000e-04, 4.000397e-10, 0.9999]),
    np.array([5.202440e-05, 2.081083e-10, 0.999948]),
    np.array([5.201061e-06, 2.080435e-11, 0.9999948]),
    np.array([5.258603e-07, 2.103442e-12, 0.9999995]),
    np.array([6.934511e-08, 2.773804e-13, 0.9999999])
]

for y in ys:
    np.testing.assert_allclose(y, expect, rtol=rtol, atol=atol) 
