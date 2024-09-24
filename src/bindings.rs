use super::{MODULE_NAME, M};

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
use pyoil3::{
    pyoil3_class,
    pyoil3_ref_class,
};
use crate::convert::{
    vec_v_to_pyarray,
    vec_t_to_pyarray,
};

type V = <M as diffsol::matrix::MatrixCommon>::V;
type T = <M as diffsol::matrix::MatrixCommon>::T;
type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;

// Aliases for diffsol classes, by type where necessary
type Context = diffsol::DiffSlContext<M>;
type Problem<'a> = diffsol::OdeSolverProblem<Eqn<'a>>;
type Builder = diffsol::OdeBuilder;

/// BDF types, static lifetime required to bypass compile time lifetime checks
type BdfCallable<'a> = diffsol::op::bdf::BdfCallable<Eqn<'a>>;
type Bdf<'a> = diffsol::Bdf::<
    M,
    Eqn<'a>,
    diffsol::NewtonNonlinearSolver<
        BdfCallable<'a>,
        <M as diffsol::DefaultSolver>::LS::<BdfCallable<'a>>
    >
>;

/* // FIXME review types
type SdirkCallable<'a> = diffsol::op::sdirk::SdirkCallable<Eqn<'a>>;
type Sdirk<'a> = diffsol::Sdirk::<
    M,
    diffsol::linear_solver::FaerLU<SdirkCallable<'a>>,
    Eqn<'a>
>; */

// Wrapped ref cells for diffsol classes that require mutation
type BuilderRefCell = RefCell<diffsol::OdeBuilder>;
type ProblemRefCell = RefCell<diffsol::OdeSolverProblem<M>>;
type BdfRefCell = RefCell<Bdf<'static>>;


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
pyoil3_ref_class!("Problem", Problem, pyoil_problem, pyoil_context);


// -----------------------------------------------------------------------------
// Builder class
// -----------------------------------------------------------------------------
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
pyoil3_class!("Bdf", BdfRefCell, pyoil_bdf);

#[pymethods]
impl pyoil_bdf::PyClass {
    #[new]
    pub fn new() -> pyoil_bdf::PyClass {
        pyoil_bdf::PyClass::bind_instance(
            RefCell::new(Bdf::default())
        )
    }

    pub fn step<'py>(
        slf: PyRefMut<'py, Self>
    ) -> PyResult<SolverStopReason> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();
        let result = solver.step();
        let state = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(SolverStopReason::from(state))
    }

    pub fn solve<'py>(
        slf: PyRefMut<'py, Self>,
        problem: &pyoil_problem::PyClass,
        final_time: Option<T>
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();

        let problem_guard = problem.0.lock().unwrap();
        let final_time = final_time.unwrap_or(1.0);
        let result = solver.solve(&problem_guard.ref_static, final_time);
        let (y, t) = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok((
            vec_v_to_pyarray::<M>(slf.py(), &y),
            vec_t_to_pyarray(slf.py(), &t),
        ))
    }

    fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        problem: &pyoil_problem::PyClass,
        t_eval: Vec<T>
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();
        let problem_guard = problem.0.lock().unwrap();
        let result = solver.solve_dense(&problem_guard.ref_static, &t_eval);
        let values = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        let pyarray = vec_v_to_pyarray::<M>(slf.py(), &values);
        Ok(pyarray)
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
    m.add_class::<pyoil_bdf::PyClass>()?;

    // Module docstring has to be programatic because #[pymodule] not used
    m.setattr("__doc__", format!("Wrapper for {} diffsol type", &MODULE_NAME))?;
    parent_module.add_submodule(&m)
}