use std::cell::RefCell;

use diffsol::{
    Bdf,
    NewtonNonlinearSolver,
    NalgebraLU,
    OdeSolverMethod,
};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyoil3::pyoil3_class;

use crate::core::types;
use crate::problem::pyoil_problem;
use crate::solution::PyOdeSolution;

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

    ///
    pub fn solve<'p>(
        slf: PyRefMut<'p, Self>,
        problem: &pyoil_problem::PyClass
    ) -> PyResult<PyOdeSolution> {
        let solver_guard = slf.0.lock().unwrap();
        let mut solver = solver_guard.instance.borrow_mut();

        let problem_guard = problem.0.lock().unwrap();
        let result = solver.solve(&problem_guard.ref_static, 1.0);
        let solution = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyOdeSolution(solution))
    }
}
