use super::{MODULE_NAME, Matrix, LinearSolver};

use std::cell::RefCell;
use pyo3::{
    prelude::*,
    Bound,
    exceptions::PyValueError,
};
use diffsol::OdeSolverMethod;
use numpy::{
    PyArray1,
    PyArray2,
};
use crate::pyoil3_class;
use crate::convert::{
    vec_v_to_pyarray,
    vec_t_to_pyarray,
};
use crate::solver_class;

type M = Matrix;
type LS<Op> = LinearSolver<Op>;
type V = <M as diffsol::matrix::MatrixCommon>::V;
type T = <M as diffsol::matrix::MatrixCommon>::T;
type DM = <V as diffsol::DefaultDenseMatrix>::M;
type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;

// Aliases for diffsol classes, by type where necessary
type Context = diffsol::DiffSlContext<M>;
type Problem<'a> = diffsol::OdeSolverProblem<Eqn<'a>>;
type Builder = diffsol::OdeBuilder;


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
type ProblemRefCell = RefCell<diffsol::OdeSolverProblem<M>>;

pyoil3_ref_class!("Problem", Problem, pyoil_problem, pyoil_context);


// -----------------------------------------------------------------------------
// Builder class
// -----------------------------------------------------------------------------
type BuilderRefCell = RefCell<diffsol::OdeBuilder>;

pyoil3_class!("Builder", BuilderRefCell, pyoil_builder);

// Helper for fluent builder interface to apply `tx` function to underlying
// diffsol class. Takes the original value, applies `tx` and replaces with new.
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
// SolverStopReason class
// -----------------------------------------------------------------------------
#[pyclass(name = "SolverStopReason")]
pub enum SolverStopReason {
    InternalTimestep { },
    RootFound { root: T },
    TstopReached { },
}

type SolverStopReasonT = diffsol::OdeSolverStopReason<T>;
impl From<SolverStopReasonT> for SolverStopReason {
    fn from(value: SolverStopReasonT) -> Self {
        match value {
            SolverStopReasonT::InternalTimestep => SolverStopReason::InternalTimestep { },
            SolverStopReasonT::RootFound(root) => SolverStopReason::RootFound { root },
            SolverStopReasonT::TstopReached => SolverStopReason::TstopReached { },
        }
    }
}


// -----------------------------------------------------------------------------
// Bdf solver class
// -----------------------------------------------------------------------------
type BdfCallable<'a> = diffsol::op::bdf::BdfCallable<Eqn<'a>>;
type BdfNonLinearSolver<'a> = diffsol::NewtonNonlinearSolver<
    BdfCallable<'a>,
    LS<BdfCallable<'a>>
>;
type Bdf<'a> = diffsol::Bdf::<
    DM,
    Eqn<'a>,
    BdfNonLinearSolver<'a>
>;
type BdfRefCell = RefCell<Bdf<'static>>;

fn bdf_constructor<'a>() -> Bdf<'a> {
    let linear_solver = LS::default();
    let nonlinear_solver = diffsol::NewtonNonlinearSolver::new(linear_solver);
    Bdf::new(nonlinear_solver)
}

solver_class!("Bdf", BdfRefCell, pyoil_bdf, bdf_constructor);


// -----------------------------------------------------------------------------
// SDIRK solver class
// -----------------------------------------------------------------------------
type SdirkCallable<'a> = diffsol::op::sdirk::SdirkCallable<Eqn<'a>>;
type Sdirk<'a> = diffsol::Sdirk::<
    DM,
    Eqn<'a>,
    LS::<SdirkCallable<'a>>
>;
type SdirkRefCell = RefCell<Sdirk<'static>>;

fn sdirk_constructor<'a>() -> Sdirk<'a> {
    let tableau = diffsol::Tableau::<DM>::tr_bdf2();
    diffsol::Sdirk::new(tableau, LS::default())
}

solver_class!("Sdirk", SdirkRefCell, pyoil_sdirk, sdirk_constructor);


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
    m.add_class::<pyoil_bdf::PyClass>()?;
    m.add_class::<pyoil_sdirk::PyClass>()?;

    // Main docstring coded rather than /// comment because #[pymodule] not used
    m.setattr("__doc__", format!("Wrapper for {} diffsol type", &MODULE_NAME))?;
    parent_module.add_submodule(&m)
}