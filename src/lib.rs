use pyo3::prelude::*;

#[macro_use]
mod py_class;

mod solver_class;
mod create_binding;
mod nalgebra_py;
mod faer_py;

// nalgebra_dense_lu_f64 binding
type MatrixNaLu = nalgebra::DMatrix<f64>;
type SolverNaLu<Op> = ::diffsol::NalgebraLU<f64, Op>;
create_binding!(nalgebra_dense_lu_f64, MatrixNaLu, SolverNaLu, nalgebra_py);

// faer_sparse_lu_f64 binding
type MatrixFaLu = ::diffsol::SparseColMat<f64>;
type SolverFaLu<Op> = ::diffsol::FaerSparseLU<f64, Op>;
create_binding!(faer_sparse_lu_f64, MatrixFaLu, SolverFaLu, faer_py);

// faer_sparse_klu_f64 binding
type MatrixFaKlu = ::diffsol::SparseColMat<f64>;
type SolverFaKlu<Op> = ::diffsol::KLU<MatrixFaKlu, Op>;
create_binding!(faer_sparse_klu_f64, MatrixFaKlu, SolverFaKlu, faer_py);

/// Top-level typed diffsol bindings
#[pymodule]
fn diffsol(m: &Bound<'_, PyModule>) -> PyResult<()> {
    nalgebra_dense_lu_f64::add_to_parent_module(m)?;
    faer_sparse_lu_f64::add_to_parent_module(m)?;
    faer_sparse_klu_f64::add_to_parent_module(m)?;
    Ok(())
}
