import pytest
import numpy as np
from tests.utils import * # ds and solver_name included here


def test_no_state(ds, solver_name):
    # Handle to state is valid even when solver not fully initialised
    state = new_solver(ds, solver_name).state()

    # Check getter/setter access exception on a subset of state variables
    with pytest.raises(Exception, match=f"{solver_name} solver has no state for getter"):
        _ = state.y

    with pytest.raises(Exception, match=f"{solver_name} solver has no state for setter"):
        state.h = 1


def test_state_getters(ds, solver_name):
    solver = new_solver(ds, solver_name)
    solver.solve(exp_decay(ds))
    state = solver.state()

    assert len(state.y) == 1
    assert len(state.dy) == 1
    assert len(state.s) == 0
    assert len(state.ds) == 0

    assert state.y[0] == pytest.approx(0.905, abs=1e-3)
    assert state.dy[0] == pytest.approx(-0.090, abs=1e-3)
    assert state.h == pytest.approx(0.086 if solver_name == "Bdf" else 0.305, abs=1e-3)
    assert state.t == 1.0


def test_state_setters(ds, solver_name):
    solver = new_solver(ds, solver_name)
    solver.solve(exp_decay(ds)) # FIXME replace exp_delay with minimal constructor
    state = solver.state()

    state.y = np.array([1, 2], dtype=np.float64)
    np.testing.assert_equal(state.y, [1, 2])

    state.dy = np.array([3, 4], dtype=np.float64)
    np.testing.assert_equal(state.dy, [3, 4])

    state.s = [np.array([5], dtype=np.float64)]
    np.testing.assert_equal(state.s[0], [5])

    state.ds = [np.array([6], dtype=np.float64)]
    np.testing.assert_equal(state.ds[0], [6])

    state.h = 7
    assert state.h == 7

    state.t = 8
    assert state.t == 8

