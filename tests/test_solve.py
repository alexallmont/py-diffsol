import numpy as np
import diffsol

def test_logistic_growth():
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
    ys, ts = solver.solve(problem)
    for t, y in zip(ts, ys):
        expect = k*y0/(y0 + (k - y0)*np.exp(-r*t))
        err = np.abs(y[0] - expect)
        assert err < 1e-6


def test_robertson_ode():
    # Test reworked from diffsol/src/ode_solver/test_models/robertson_ode.rs
    # 3-species kinetics problem
    #    dy1/dt = -.04*y1 + 1.e4*y2*y3
    #    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
    #    dy3/dt = 3.e7*(y2)^2
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
    t_eval = [0.0, 0.4, 4.0, 40.0, 400.0, 4000.0, 40000.0, 400000.0, 4000000.0, 4.0e7, 4.0e8, 4.0e9, 4.0e10]
    ys = solver.solve_dense(problem, t_eval)

    expected = [
        np.array([1.0, 0.0, 0.0]),
        np.array([9.851641e-01, 3.386242e-05, 1.480205e-02]),
        np.array([9.055097e-01, 2.240338e-05, 9.446793e-02]),
        np.array([7.158017e-01, 9.185037e-06, 2.841892e-01]),
        np.array([4.505360e-01, 3.223271e-06, 5.494608e-01]),
        np.array([1.832299e-01, 8.944378e-07, 8.167692e-01]),
        np.array([3.898902e-02, 1.622006e-07, 9.610108e-01]),
        np.array([4.936383e-03, 1.984224e-08, 9.950636e-01]),
        np.array([5.168093e-04, 2.068293e-09, 9.994832e-01]),
        np.array([5.202440e-05, 2.081083e-10, 9.999480e-01]),
        np.array([5.201061e-06, 2.080435e-11, 9.999948e-01]),
        np.array([5.258603e-07, 2.103442e-12, 9.999995e-01]),
        np.array([6.934511e-08, 2.773804e-13, 9.999999e-01])
    ]

    rtol = 1e-4
    atol = 1e-8
    np.testing.assert_allclose(ys, expected, rtol=rtol, atol=atol)
