mod builder;
mod problem;

use crate::builder::PyOdeBuilder;
use crate::problem::PyOdeSolverProblem;

use::pyo3::prelude::*;

#[pymodule]
fn diffsol<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<PyOdeBuilder>()?;
    Ok(())
}