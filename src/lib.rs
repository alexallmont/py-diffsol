use pyo3::prelude::*;

#[path = "."]
mod nalgebra_dense_f64 {
    static MODULE_NAME:&'static str = "nalgebra_dense_f64";
    type M = nalgebra::DMatrix<f64>;
    mod bindings;
    pub use bindings::*;
}

#[path = "."]
mod faer_sparse_f64 {
    static MODULE_NAME:&'static str = "faer_sparse_f64";
    type M = diffsol::SparseColMat<f64>;
    mod bindings;
    pub use bindings::*;
}

/// Top-level typed diffsol bindings
#[pymodule]
fn diffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nalgebra_dense_f64::add_to_parent_module(m)?;
    faer_sparse_f64::add_to_parent_module(m)?;
    Ok(())
}
