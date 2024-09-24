use pyo3::prelude::*;

mod convert;

#[cfg(feature = "nalgebra_dense_lu_f64")]
#[path = "."]
mod nalgebra_dense_lu_f64 {
    static MODULE_NAME:&'static str = "nalgebra_dense_lu_f64";
    type Matrix = nalgebra::DMatrix<f64>;
    type LinearSolver<Op> = diffsol::NalgebraLU<f64, Op>;
    mod bindings;
    pub use bindings::*;
}

#[cfg(feature = "faer_sparse_lu_f64")]
#[path = "."]
mod faer_sparse_lu_f64 {
    static MODULE_NAME:&'static str = "faer_sparse_lu_f64";
    type Matrix = diffsol::SparseColMat<f64>;
    type LinearSolver<Op> = diffsol::FaerSparseLU<f64, Op>;
    mod bindings;
    pub use bindings::*;
}

#[cfg(feature = "faer_sparse_klu_f64")]
#[path = "."]
mod faer_sparse_klu_f64 {
    static MODULE_NAME:&'static str = "faer_sparse_klu_f64";
    type Matrix = diffsol::SparseColMat<f64>;
    type LinearSolver<Op> = diffsol::KLU<Matrix, Op>;
    mod bindings;
    pub use bindings::*;
}

/// Top-level typed diffsol bindings
#[pymodule]
fn diffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {

    #[cfg(feature = "nalgebra_dense_lu_f64")]
    nalgebra_dense_lu_f64::add_to_parent_module(m)?;

    #[cfg(feature = "faer_sparse_lu_f64")]
    faer_sparse_lu_f64::add_to_parent_module(m)?;

    #[cfg(feature = "faer_sparse_klu_f64")]
    faer_sparse_klu_f64::add_to_parent_module(m)?;

    Ok(())
}
