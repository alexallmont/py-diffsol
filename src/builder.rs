use std::cell::RefCell;

use diffsol::OdeBuilder;
use pyo3::prelude::*;
use pyoil3::pyoil3_class;

use crate::problem::pyoil_problem;

// 
type OdeBuilderRefCell = RefCell<OdeBuilder>;
pyoil3_class!(
    "OdeBuilder",
    OdeBuilderRefCell,
    pyoil_builder
);

// Call an OdeBuilder builder method on the container object. This allows for a builder/fluent
// mechanism in PyO3 to an underlying builder class by taking the builder object, operating on
// it with the `tx` lambda, and then replacing the original RefCell with the builder response.
fn _apply_builder_fn<F: Fn(OdeBuilder) -> OdeBuilder>(
    builder: &pyoil_builder::ArcHandle,
    tx: F
) {
    let cl = builder.clone();
    let builder = cl.lock().unwrap();
    let transformed = tx(builder.instance.take());
    builder.instance.replace(transformed);
}


/// Public OdeBuilder python methods
#[pymethods]
impl pyoil_builder::PyClass {
    #[new]
    pub fn new() -> Self {
        Self::bind_instance(RefCell::new(OdeBuilder::new()))
    }

    /// Set the initial time.
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

    /// Set the initial step size.
    pub fn h0<'p>(slf: PyRefMut<'p, Self>, h0: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.h0(h0));
        slf
    }

    /// Set the relative tolerance.
    pub fn rtol<'p>(slf: PyRefMut<'p, Self>, rtol: f64) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.rtol(rtol));
        slf
    }

    /// Set the absolute tolerance.
    pub fn atol<'p>(slf: PyRefMut<'p, Self>, atol: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.atol(atol.to_vec()));
        slf
    }

    /// Set the parameters.
    pub fn p<'p>(slf: PyRefMut<'p, Self>, p: Vec<f64>) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.p(p.to_vec()));
        slf
    }

    /// Set whether to use coloring when computing the Jacobian.
    pub fn use_coloring<'p>(slf: PyRefMut<'p, Self>, use_coloring: bool) -> PyRefMut<'p, Self> {
        _apply_builder_fn(&slf.0, |t| t.use_coloring(use_coloring));
        slf
    }

    /// Build an OdeSolverProblem from a diffsl code string given a context
    pub fn build_diffsl<'p>(
        slf: PyRefMut<'p, Self>,
        context: PyRef<'p, crate::context::pyoil_context::PyClass>
    ) -> PyResult<pyoil_problem::PyClass> {
        // FIXME replace unwraps
        let guard = slf.0.lock().unwrap();
        let instance = &context.0.lock().unwrap().instance;
        let builder = guard.instance.take();
        let problem = builder.build_diffsl(&instance).unwrap();

        let result = pyoil_problem::PyClass::bind_owned_instance(
            problem,
            context.0.clone()
        );
        Ok(result)
    }
}
