use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::sync::{
    Arc,
    Mutex,
};

use diffsol::{
    DiffSlContext,
    OdeSolverProblem,
};

// FIXME
// - move context to separate file
// - wrap OdeSolverState
// - remove pub access
// - review multithreaded access. Is sync/send safe?

/// Internal storage for a public PyOdeSolverContext object.
/// 
/// This is necessary to implement null Send and Sync methods, which informs Rust
/// that we are taking responsibility of any threading issues, and not to raise
/// errors about the internals of diffsol (in particular Rcs in the library and 
/// LLVMBuilderRef in Enzyme). These will not be an issue because the public API
/// is hidden behind an Arc Mutex, so will never be accessed concurrently.
pub struct OdeSolverContextInstance {
    pub context: DiffSlContext,
}
unsafe impl Send for OdeSolverContextInstance {}
unsafe impl Sync for OdeSolverContextInstance {}

#[pyclass]
#[pyo3(name = "OdeSolverContext")]
pub struct PyOdeSolverContext(pub Arc<Mutex<OdeSolverContextInstance>>);

#[pymethods]
impl PyOdeSolverContext {
    /// Create a new context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
    /// This contains the compiled code and the data structures needed to evaluate the ODE equations.
    #[new]
    pub fn new(code: &str) -> PyResult<PyOdeSolverContext> {
        // Try creating a native diffsl context from the code, and wrap in PyOdeSolverContext
        let wrapped_context = DiffSlContext::new(code); // FIXME address exception in parse_ds_string in compiler.rs then use ? instead.
        match wrapped_context {
            Ok(context) => {
                Ok(
                    PyOdeSolverContext(Arc::new(Mutex::new(
                        OdeSolverContextInstance{
                            context: context,
                        }
                    )))
                )
            },
            Err(err) => {
                Err(PyValueError::new_err(err.to_string()))
            }
        }
    }
}

/// Internal storage for a public PyOdeSolverProblem object.
/// 
/// This class keeps an Arc clone of the owning context that created the problem,
/// which increases the Arc::strong_count and so ensure the context will not be
/// dropped until after the lifetime of the problem. This is required because
/// OdeSolverProblem has a compile-time lifetime dependency on the DiffSlContext
/// instance (see return type of OdeBuilder::build_diffsl).
/// To circumvent the compile-time lifetime, the problem is stored as 'static
/// and then transmuted to and from the lifetime of the context.
struct OdeSolverProblemInstance {
    owning_context: PyOdeSolverContext,
    problem_static: OdeSolverProblem<diffsol::ode_solver::diffsl::DiffSl<'static>>,
}
unsafe impl Send for OdeSolverProblemInstance {}
unsafe impl Sync for OdeSolverProblemInstance {}

#[pyclass]
#[pyo3(name = "OdeSolverProblem")]
pub struct PyOdeSolverProblem(Arc<Mutex<OdeSolverProblemInstance>>);

// Non-python methods
impl PyOdeSolverProblem {
    pub fn create_problem_handle<'a>(
        owning_context: &PyOdeSolverContext,
        _context_ref: &'a DiffSlContext,
        problem: OdeSolverProblem<diffsol::ode_solver::diffsl::DiffSl<'a>>
    ) -> PyOdeSolverProblem {
        let problem_static: OdeSolverProblem<diffsol::ode_solver::diffsl::DiffSl<'static>> = unsafe { std::mem::transmute(problem) };
        PyOdeSolverProblem(Arc::new(Mutex::new(
            OdeSolverProblemInstance{
                owning_context: PyOdeSolverContext(owning_context.0.clone()),
                problem_static: problem_static,
            }
        )))
    }
}
