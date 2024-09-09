//! Common types required in project
//!
//! Diffsol has flexible generic types that can be determined at compile time.
//! However, python is purely runtime so some of the type information must be
//! specified up-front. In particular the matrix storage and their underlying
//! element type of f64.

pub mod nalgebra_dense {
    pub type T = f64;
    pub type M = nalgebra::DMatrix<T>;
    pub type V = <M as diffsol::matrix::MatrixCommon>::V;
    pub type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;
}

pub mod faer_sparse {
    pub type T = f64;
    pub type M = diffsol::SparseColMat<T>;
    pub type V = <M as diffsol::matrix::MatrixCommon>::V;
    pub type Eqn<'a> = diffsol::ode_solver::diffsl::DiffSl<'a, M>;
}

// FIXME testing faer but these types need to be runtime-swappable
pub use nalgebra_dense as types;