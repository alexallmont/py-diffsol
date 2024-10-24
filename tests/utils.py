import pytest

from diffsol import nalgebra_dense_lu_f64 as _ds_ndl
from diffsol import faer_sparse_lu_f64 as _ds_fsl
from diffsol import faer_sparse_klu_f64 as _ds_fsk

@pytest.fixture(params=["Bdf", "Sdirk"])
def solver_name(request):
    """
    Fixture parameterizing all solvers

    This avoids code duplication per test. For example rather than
        @pytest.mark.parametrize("ds", [ds_ndl, ds_fsl, ds_fsk])
        @pytest.mark.parametrize("solver_name", ["Bdf", "Sdirk"])
        def test_foo(ds, solver_names):
            solver = new_solver(ds, solver_name)
            ...

    the code is reduced to
        from common import *
        def test_foo(ds, solver_names):
            solver = new_solver(ds, solver_name)
    """
    return request.param


@pytest.fixture(params=[_ds_ndl, _ds_fsl, _ds_fsk])
def ds(request):
    """
    Fixture parameterizing all typed DiffSl systems

    See solver_name() for reference on usage.
    """
    return request.param


def new_solver(ds, solver_name):
    """
    Create a new solver given diffsol module an name of the solver

    For example if ds is nalgebra_dense_lu_f64 and solver_name is "Bdf", this is
    equivalent to nalgebra_dense_lu_f64.Bdf().
    """
    solver_class = getattr(ds, solver_name)
    return solver_class()


def exp_decay(ds):
    """
    Create a new exponential decay problem for given diffsol module
    """
    context = ds.Context("in = [a] a { 1 } u { 1.0 } F { -a*u } out { u }")
    builder = ds.Builder().rtol(1e-6).p([0.1]).h0(5.0)
    return builder.build_diffsl(context)
