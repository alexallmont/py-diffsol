use std::cell::RefCell;

use diffsol::{
    Bdf,
    NalgebraLU,
    NewtonNonlinearSolver,
    OdeSolverMethod
};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyoil3::pyoil3_class;

use crate::core::types;
use crate::problem::pyoil_problem;
use crate::solution::{BoundPyArray2, PyOdeSolution, vec_v_to_pyarray};
use crate::stop_reason::PyOdeSolverStopReason;

/// Types specific to BDF solver
pub type Callable<'a> = diffsol::op::bdf::BdfCallable<types::Eqn<'a>>;
pub type DefaultBdf<'a> = Bdf::<
    types::M,
    types::Eqn<'a>,
    NewtonNonlinearSolver<Callable<'a>, NalgebraLU<types::T, Callable<'a>>>
>;

/// pyoil_bdf python wrapper for DefaultBdf instance. Static lifetime required
/// to bypass compile time lifetime checks.
type DefaultBdfRefCell = RefCell<DefaultBdf<'static>>;

pyoil3_class!(
    "Bdf",
    DefaultBdfRefCell,
    pyoil_bdf
);

#[pymethods]
impl pyoil_bdf::PyClass {
    #[new]
    pub fn new() -> pyoil_bdf::PyClass {
        pyoil_bdf::PyClass::bind_instance(
            RefCell::new(DefaultBdf::default())
        )
    }

    // TODO fn problem(&self) -> Option<&OdeSolverProblem<Eqn>>;
    // TODO fn set_problem(&mut self, state: OdeSolverState<Eqn::V>, problem: &OdeSolverProblem<Eqn>);

    /// TODO docs import
    pub fn step<'p>(
        slf: PyRefMut<'p, Self>
    ) -> PyResult<PyOdeSolverStopReason> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();
        let result = solver.step();
        let state = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyOdeSolverStopReason::from(state))
    }

    // TODO fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<()>;
    // TODO fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V>;
    // TODO fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>>;
    // TODO fn state(&self) -> Option<&OdeSolverState<Eqn::V>>;
    // TODO fn order(&self) -> usize;

    /// TODO docs import
    pub fn solve<'p>(
        slf: PyRefMut<'p, Self>,
        problem: &pyoil_problem::PyClass,
        final_time: Option<types::T>
    ) -> PyResult<PyOdeSolution> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();

        let problem_guard = problem.0.lock().unwrap();
        let final_time = final_time.unwrap_or(1.0);
        let result = solver.solve(&problem_guard.ref_static, final_time);
        let solution = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyOdeSolution(solution))
    }

    /// TODO docs import
    pub fn solve_dense<'py>(
        slf: PyRefMut<'py, Self>,
        problem: &pyoil_problem::PyClass,
        t_eval: Vec<types::T>
    ) -> PyResult<BoundPyArray2<'py>> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();

        let problem_guard = problem.0.lock().unwrap();
        let result = solver.solve_dense(&problem_guard.ref_static, &t_eval);
        let values = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        let pyarray = vec_v_to_pyarray(slf.py(), &values);

        Ok(pyarray)
    }
}
