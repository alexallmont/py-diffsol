use diffsol::DiffSlContext;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use pyoil3::pyoil3_class;

pyoil3_class!(
    "OdeSolverContext",
    DiffSlContext,
    pyoil_context
);

#[pymethods]
impl pyoil_context::PyClass {
    #[new]
    pub fn new(code: &str) -> PyResult<Self> {
        // Try creating a native diffsl context from the code, and wrap in PyOdeSolverContext
        // FIXME DiffSlContext::new can panic in parse_ds_string; change diffsol
        // to use result and use ? here instead.
        let context = DiffSlContext::new(code);
        let inst = context.map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(pyoil_context::PyClass::bind_instance(inst))
    }
}
