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

/// FIXME remove work in progress for running python from Rust.
#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_run_python() {
        pyo3::prepare_freethreaded_python();
        Python::with_gil(|py| {
            let sys = py.import_bound("sys").unwrap();
            let version: String = sys.getattr("version").unwrap().extract().unwrap();

            let locals = [("os", py.import_bound("os").unwrap())].into_py_dict_bound(py);
            let code = "os.getenv('USER') or os.getenv('USERNAME') or 'Unknown'";
            let user: String = py.eval_bound(code, None, Some(&locals)).unwrap().extract()?;

            println!("Hello {}, I'm Python {}", user, version);
            PyResult::Ok(())
        }).unwrap();
    }
}
