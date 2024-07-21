use::pyo3::prelude::*;

mod core;

mod bdf;
mod builder;
mod context;
mod problem;
mod solution;

#[pymodule]
fn diffsol<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_class::<bdf::pyoil_bdf::PyClass>()?;
    m.add_class::<builder::pyoil_builder::PyClass>()?;
    m.add_class::<context::pyoil_context::PyClass>()?;
    m.add_class::<problem::pyoil_problem::PyClass>()?;
    m.add_class::<solution::PyOdeSolution>()?;
    Ok(())
}
