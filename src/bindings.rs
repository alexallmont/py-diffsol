use super::{MODULE_NAME, M};

use std::cell::RefCell;
use pyo3::{
    prelude::*,
    Bound,
    exceptions::PyValueError
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyoil3::{pyoil3_class, pyoil3_ref_class};

type V = <M as diffsol::matrix::MatrixCommon>::V;
type T = <M as diffsol::matrix::MatrixCommon>::T;
type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;

pub type BoundPyArray1<'py> = Bound<'py, PyArray1<T>>;
pub type BoundPyArray2<'py> = Bound<'py, PyArray2<T>>;

// Aliases for diffsol classes, by type where necessary
type Context = diffsol::DiffSlContext<M>;
type Problem<'a> = diffsol::OdeSolverProblem<Eqn<'a>>;
type Builder = diffsol::OdeBuilder;

// Wrapped ref cells for diffsol classes that require mutation
type BuilderRefCell = RefCell<diffsol::OdeBuilder>;
type ProblemRefCell = RefCell<diffsol::OdeSolverProblem<M>>;

// Solver problem is aliased as pyoil3 cannot parse generics
type SolverProblem<'a> = diffsol::OdeSolverProblem<Eqn<'a>>;


// -----------------------------------------------------------------------------
// Context class
// -----------------------------------------------------------------------------
pyoil3_class!("Context", Context, pyoil_context);

#[pymethods]
impl pyoil_context::PyClass {
    #[new]
    pub fn new(code: &str) -> PyResult<Self> {
        // Create a native diffsl context from the code, and wrap in Context
        let context = diffsol::DiffSlContext::new(code);
        let inst = context.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(pyoil_context::PyClass::bind_instance(inst))
    }
}


// -----------------------------------------------------------------------------
// Problem class
// -----------------------------------------------------------------------------
pyoil3_ref_class!(
    "Problem",
    Problem,
    pyoil_problem,
    pyoil_context
);


// -----------------------------------------------------------------------------
// Builder class
// -----------------------------------------------------------------------------
pyoil3_class!(
    "Builder",
    BuilderRefCell,
    pyoil_builder
);

// Call an OdeBuilder builder method on the container object to allow the fluent
// builder class to change the underlying diffsol builder class, by operating
// `tx` on it and replacing the original RefCell with the builder response.
fn _apply_builder_fn<F: Fn(Builder) -> Builder>(
    builder: &pyoil_builder::ArcHandle,
    tx: F
) {
    let cl = builder.clone();
    let builder = cl.lock().unwrap();
    let transformed = tx(builder.instance.take());
    builder.instance.replace(transformed);
}

#[pymethods]
impl pyoil_builder::PyClass {
    #[new]
    pub fn new() -> Self {
        Self::bind_instance(RefCell::new(Builder::new()))
    }

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

    pub fn h0<'p>(slf: PyRefMut<'p, Self>, h0: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.h0(h0));
        slf
    }

    pub fn rtol<'p>(slf: PyRefMut<'p, Self>, rtol: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.rtol(rtol));
        slf
    }

    pub fn atol<'p>(slf: PyRefMut<'p, Self>, atol: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.atol(atol.to_vec()));
        slf
    }

    pub fn p<'p>(slf: PyRefMut<'p, Self>, p: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.p(p.to_vec()));
        slf
    }

    pub fn use_coloring<'p>(slf: PyRefMut<'p, Self>, use_coloring: bool) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.use_coloring(use_coloring));
        slf
    }

    pub fn build_diffsl<'p>(
        slf: PyRefMut<'p, Self>,
        context: PyRef<'p, pyoil_context::PyClass>
    ) -> PyResult<pyoil_problem::PyClass> {
        // FIXME replace unwraps
        let builder_guard = slf.0.lock().unwrap();
        let context_guard = context.0.lock().unwrap();

        let builder = builder_guard.instance.take();
        let instance = &context_guard.instance;
        let problem = builder.build_diffsl(instance).unwrap();
        let result = pyoil_problem::PyClass::bind_owned_instance(
            problem,
            context.0.clone()
        );
        Ok(result)
    }
}


// -----------------------------------------------------------------------------
// Module declaration
// -----------------------------------------------------------------------------
pub fn add_to_parent_module(
    parent_module: &Bound<'_, PyModule>
) -> PyResult<()> {
    let m = PyModule::new_bound(parent_module.py(), &MODULE_NAME)?;
    m.add_class::<pyoil_context::PyClass>()?;
    m.add_class::<pyoil_problem::PyClass>()?;
    m.add_class::<pyoil_builder::PyClass>()?;

    // Module docstring has to be programatic because #[pymodule] not used
    m.setattr("__doc__", format!("Wrapper for {} diffsol type", &MODULE_NAME))?;
    parent_module.add_submodule(&m)
}