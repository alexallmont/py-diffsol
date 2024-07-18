use std::cell::RefCell;

use diffsol::{
    Bdf,
    NewtonNonlinearSolver,
    NalgebraLU,
    OdeSolverMethod,
};
use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyoil3::pyoil3_class;

use crate::core::types;

/// Types specific to BDF solver
pub type Callable = diffsol::op::bdf::BdfCallable<types::Eqn>;
pub type DefaultBdf = Bdf::<
    types::M,
    types::Eqn,
    NewtonNonlinearSolver<Callable, NalgebraLU<types::T, Callable>>
>;

/// pyoil_bdf python wrapper for DefaultBdf instance.
type DefaultBdfRefCell = RefCell<DefaultBdf>;
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
        problem: &crate::problem::pyoil_problem::PyClass
    ) -> PyResult<pyo3::Bound<'p, PyArray2<f64>>> {
        // FIXME remove unwraps
        let solver = slf.0.lock().unwrap();
        let problem = problem.0.lock().unwrap();

        let mut solver = solver.instance.borrow_mut();
        let result = solver.solve(&problem.ref_static, 1.0);
        let Ok(matrix) = result else {
            return Err(PyValueError::new_err("Solver error".to_string())); // FIXME review error
        };

        Ok(crate::matrix::matrix_to_py(slf.py(), &matrix))
    }
}
