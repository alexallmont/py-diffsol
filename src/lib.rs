mod builder;
mod problem;

use crate::builder::PyOdeBuilder;
use crate::problem::PyOdeSolverContext;
use crate::problem::PyOdeSolverProblem;

use::pyo3::prelude::*;

#[pymodule]
fn diffsol<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyOdeBuilder>()?;
    m.add_class::<PyOdeSolverContext>()?;
    m.add_class::<PyOdeSolverProblem>()?;
    Ok(())
}