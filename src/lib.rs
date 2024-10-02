use pyo3::prelude::*;

#[macro_use]
mod pyoil3_class;

mod convert;
mod solver_class;
mod create_binding;

// nalgebra_dense_lu_f64
type MatrixNaLu = nalgebra::DMatrix<f64>;
type SolverNaLu<Op> = ::diffsol::NalgebraLU<f64, Op>;
create_binding!(nalgebra_dense_lu_f64, MatrixNaLu, SolverNaLu);

// faer_sparse_lu_f64
type MatrixFaLu = ::diffsol::SparseColMat<f64>;
type SolverFaLu<Op> = ::diffsol::FaerSparseLU<f64, Op>;
create_binding!(faer_sparse_lu_f64, MatrixFaLu, SolverFaLu);

// faer_sparse_klu_f64
type MatrixFaKlu = ::diffsol::SparseColMat<f64>;
type SolverFaKlu<Op> = ::diffsol::KLU<MatrixFaKlu, Op>;
create_binding!(faer_sparse_klu_f64, MatrixFaKlu, SolverFaKlu);

/// Top-level typed diffsol bindings
#[pymodule]
fn diffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nalgebra_dense_lu_f64::add_to_parent_module(m)?;
    faer_sparse_lu_f64::add_to_parent_module(m)?;
    faer_sparse_klu_f64::add_to_parent_module(m)?;
    Ok(())
}
