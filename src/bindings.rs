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
use crate::convert::{
    vec_v_to_pyarray,
    vec_t_to_pyarray,
};
use crate::solver_class;

/// Types for this module's binding derived from super:: settings
type M = Matrix;
type LS<Op> = LinearSolver<Op>;
type V = <M as diffsol::matrix::MatrixCommon>::V;
type T = <M as diffsol::matrix::MatrixCommon>::T;
type DM = <V as diffsol::DefaultDenseMatrix>::M;
type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;


/// Convert common diffsol errors to PyErrors
fn diffsol_err(err: diffsol::error::DiffsolError) -> PyErr {
    PyValueError::new_err(err.to_string())
}


// -----------------------------------------------------------------------------
// Context class
// -----------------------------------------------------------------------------
type Context = diffsol::DiffSlContext<M>;

py_class!("Context", Context, py_context);

#[pymethods]
impl py_context::PyClass {
    #[new]
    pub fn new(code: &str) -> PyResult<Self> {
        // Create a native diffsl context from the code, and wrap in Context
        let context = diffsol::DiffSlContext::new(code);
        let inst = context.map_err(diffsol_err)?;
        Ok(py_context::PyClass::new_binding(inst))
    }
}


// -----------------------------------------------------------------------------
// Problem class
// -----------------------------------------------------------------------------
type Problem<'a> = diffsol::OdeSolverProblem<Eqn<'a>>;

py_class_dependant!("Problem", Problem, py_problem, py_context);


// -----------------------------------------------------------------------------
// Builder class
// -----------------------------------------------------------------------------
type Builder = RefCell<diffsol::OdeBuilder>; // RefCell for take/replace below

py_class!("Builder", Builder, py_builder);

// Helper for fluent builder interface to apply `tx` function to underlying
// diffsol class. Takes the original value, applies `tx` and replaces with new.
fn apply_builder_fn<'p, TxFn>(
    builder: PyRefMut<'p, py_builder::PyClass>,
    tx_fn: TxFn
) -> PyRefMut<'p, py_builder::PyClass>
where
    TxFn: Fn(diffsol::OdeBuilder) -> diffsol::OdeBuilder
{
    builder.use_inst(|builder| {
        // Take and replace of RefCell required to support underlying OdeBuilder
        // API which consumes a value and returns a new instance.
        let new_builder = tx_fn(builder.take());
        builder.replace(new_builder);
    });

    // Return of self here for brevity below; i.e. each method does not need an
    // additional trailing `slf`, it happens here instead.
    builder
}

#[pymethods]
impl py_builder::PyClass {
    #[new]
    pub fn new() -> Self {
        Self::new_binding(RefCell::new(diffsol::OdeBuilder::new()))
    }

    pub fn t0<'p>(slf: PyRefMut<'p, Self>, t0: f64) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.t0(t0))
    }

    pub fn sensitivities<'p>(slf: PyRefMut<'p, Self>, sensitivities: bool) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.sensitivities(sensitivities))
    }

    pub fn sensitivities_error_control<'p>(slf: PyRefMut<'p, Self>, sensitivities_error_control: bool) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.sensitivities_error_control(sensitivities_error_control))
    }

    pub fn h0<'p>(slf: PyRefMut<'p, Self>, h0: f64) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.h0(h0))
    }

    pub fn rtol<'p>(slf: PyRefMut<'p, Self>, rtol: f64) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.rtol(rtol))
    }

    pub fn atol<'p>(slf: PyRefMut<'p, Self>, atol: Vec<f64>) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.atol(atol.to_vec()))
    }

    pub fn p<'p>(slf: PyRefMut<'p, Self>, p: Vec<f64>) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.p(p.to_vec()))
    }

    pub fn use_coloring<'p>(slf: PyRefMut<'p, Self>, use_coloring: bool) -> PyRefMut<'p, Self> {
        apply_builder_fn(slf, |t| t.use_coloring(use_coloring))
    }

    pub fn build_diffsl<'p>(
        slf: PyRefMut<'p, Self>,
        context: PyRef<'p, py_context::PyClass>
    ) -> PyResult<py_problem::PyClass> {
        slf.use_inst(|builder| {
            context.use_inst(|ode_context| {
                let ode_builder = builder.take();
                let problem = ode_builder.build_diffsl(ode_context).unwrap();
                let result = py_problem::PyClass::new_binding(
                    problem,
                    context.0.clone()
                );
                Ok(result)
            })
        })
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
// Bdf requires a compile-time type definition rather than using Bdf::default()
// which is idiomatic for diffsol using Rust's type dependencies. PyO3 needs a
// pre-determined type to work with. Solvers are wrapped in RefCells because the
// OdeSolverMethods require a mutable borrow. See `solver_class` macro for
// implementation that exposes the common methods to python, with edge cases
// like Bdf or Sdirk constructor function passed in as macro arguments.
type BdfCallable<'a> = diffsol::op::bdf::BdfCallable<Eqn<'a>>;
type BdfNonLinearSolver<'a> = diffsol::NewtonNonlinearSolver<
    BdfCallable<'a>,
    LS<BdfCallable<'a>>
>;
type BdfType<'a> = diffsol::Bdf::<
    DM,
    Eqn<'a>,
    BdfNonLinearSolver<'a>
>;
type Bdf = RefCell<BdfType<'static>>; // RefCell for mutable borrows in calls

fn bdf_constructor<'a>() -> BdfType<'a> {
    let linear_solver = LS::default();
    let nonlinear_solver = diffsol::NewtonNonlinearSolver::new(linear_solver);
    BdfType::new(nonlinear_solver)
}

solver_class!("Bdf", Bdf, py_bdf, bdf_constructor);


// -----------------------------------------------------------------------------
// SDIRK solver class
// -----------------------------------------------------------------------------
// Like Bdf, Sdirk requires a compile-time type rather than idiomatic diffsol
// default() for PyO3 needing a pre-determined type, and is wrapped in RefCell
// and has a custom constructor.
type SdirkCallable<'a> = diffsol::op::sdirk::SdirkCallable<Eqn<'a>>;
type SdirkType<'a> = diffsol::Sdirk::<
    DM,
    Eqn<'a>,
    LS::<SdirkCallable<'a>>
>;
type Sdirk = RefCell<SdirkType<'static>>; // RefCell for mutable borrows in calls

fn sdirk_constructor<'a>() -> SdirkType<'a> {
    let tableau = diffsol::Tableau::<DM>::tr_bdf2();
    diffsol::Sdirk::new(tableau, LS::default())
}

solver_class!("Sdirk", Sdirk, py_sdirk, sdirk_constructor);


// -----------------------------------------------------------------------------
// Module declaration
// -----------------------------------------------------------------------------
pub fn add_to_parent_module(
    parent_module: &Bound<'_, PyModule>
) -> PyResult<()> {
    let m = PyModule::new_bound(parent_module.py(), &MODULE_NAME)?;

    m.add_class::<py_context::PyClass>()?;
    m.add_class::<py_problem::PyClass>()?;
    m.add_class::<py_builder::PyClass>()?;
    m.add_class::<py_bdf::PyClass>()?;
    m.add_class::<py_sdirk::PyClass>()?;

    // Main docstring coded rather than /// comment because #[pymodule] not used
    m.setattr("__doc__", format!("Wrapper for {} diffsol type", &MODULE_NAME))?;
    parent_module.add_submodule(&m)
}