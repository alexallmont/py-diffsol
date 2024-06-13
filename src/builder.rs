use std::cell::RefCell;
use std::sync::{
    Arc,
    Mutex
};

use diffsol::{
    DiffSlContext,
    OdeBuilder,
    OdeSolverState,
    OdeSolverMethod,
    Bdf
};
use numpy::PyArray2;
use ndarray::Array2;
use pyo3::{
    prelude::*,
    exceptions::PyRuntimeError
};


/// Builder for ODE problems. Use methods to set parameters and then call one of the build methods when done.
// Implementation note: this uses RefCell to provide interior mutability in builder pattern; it circumvents
// strictness in PyO3 when modifying and returning self.
#[pyclass]
#[pyo3(name = "OdeBuilder")]
pub struct PyOdeBuilder(Arc<Mutex<RefCell<OdeBuilder>>>);


// Call an OdeBuilder builder method on the container object. This allows for a builder/fluent
// mechanism in PyO3 to an underlying builder class by taking the builder object, operating on
// it with the `tx` lambda, and then replacing the original RefCell with the builder response.
fn _apply_builder_fn<F: Fn(OdeBuilder) -> OdeBuilder>(
    builder: &Arc<Mutex<RefCell<OdeBuilder>>>,
    tx: F
) {
    let cl = builder.clone();
    let val = cl.lock().unwrap();
    val.replace(tx(val.take()));
}


/// Public OdeBuilder python methods
#[pymethods]
impl PyOdeBuilder {
    #[new]
    pub fn new() -> PyOdeBuilder {
        PyOdeBuilder(Arc::new(Mutex::new(RefCell::new(OdeBuilder::new()))))
    }

    /// Set the initial time.
    pub fn t0<'p>(slf: PyRefMut<'p, Self>, t0: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.t0(t0));
        slf
    }

    pub fn sensitivities<'p>(slf: PyRefMut<'p, Self>, sensitivities: bool) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.sensitivities(sensitivities));
        slf
    }

    pub fn sensitivities_error_control<'p>(slf: PyRefMut<'p, Self>, sensitivities_error_control: bool) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.sensitivities_error_control(sensitivities_error_control));
        slf
    }

    /// Set the initial step size.
    pub fn h0<'p>(slf: PyRefMut<'p, Self>, h0: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.h0(h0));
        slf
    }

    /// Set the relative tolerance.
    pub fn rtol<'p>(slf: PyRefMut<'p, Self>, rtol: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.rtol(rtol));
        slf
    }

    /// Set the absolute tolerance.
    pub fn atol<'p>(slf: PyRefMut<'p, Self>, atol: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.atol(atol.to_vec()));
        slf
    }

    /// Set the parameters.
    pub fn p<'p>(slf: PyRefMut<'p, Self>, p: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.p(p.to_vec()));
        slf
    }

    /// Set whether to use coloring when computing the Jacobian.
    pub fn use_coloring<'p>(slf: PyRefMut<'p, Self>, use_coloring: bool) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.use_coloring(use_coloring));
        slf
    }

    /// Build an OdeSolverProblem from a diffsl code string
    pub fn build_diffsl<'p>(
        slf: PyRefMut<'p, Self>,
        code: &str
    ) -> PyResult<isize> {
        let guard = slf.0.lock().unwrap();
        let builder = guard.take();
        let context = DiffSlContext::new(code).unwrap();
        let _problem = builder.build_diffsl(&context).unwrap();
        Ok(-1)
    }

    /// WIP
    pub fn wip_diffsl_solve<'p>(
        slf: PyRefMut<'p, Self>,
        code: &str
    ) -> PyResult<PyObject> {
        let guard = slf.0.lock().unwrap();
        let builder = guard.take();
        let context = DiffSlContext::new(code).unwrap();
        let problem = builder.build_diffsl(&context).unwrap();

        // WIP from diffsol ode_solver/diffsl.rs
        let mut solver = Bdf::default();
        let t = 0.4;
        let state = OdeSolverState::new(&problem, &solver).unwrap();
        solver.set_problem(state, &problem);
        while solver.state().unwrap().t <= t {
            solver.step().unwrap();
        }
        if let Ok(y) = solver.interpolate(t) {
            let rows = y.nrows();
            let cols = y.ncols();
            let vec: Vec<f64> = y.iter().cloned().collect();
            let array = Array2::from_shape_vec((rows, cols), vec).unwrap();
            let py_array = PyArray2::from_owned_array_bound(slf.py(), array);
            Ok(py_array.into_py(slf.py()).to_owned())
        }
        else {
            Err(PyRuntimeError::new_err("FIXME"))
        }
    }
}
