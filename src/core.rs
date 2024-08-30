//! Common types required in project
//!
//! Diffsol has flexible generic types that can be determined at compile time.
//! However, python is purely runtime so some of the type information must be
//! specified up-front. In particular the matrix storage and their underlying
//! element type of f64.

pub mod types {
    use diffsol::ode_solver::diffsl;

    /// Underlying value and matrix types.
    // FIXME add faer::Mat and matrix::sparse_faer::SparseColMat
    pub type T = f64;
    pub type M = nalgebra::DMatrix<T>;
    pub type V = <M as diffsol::matrix::MatrixCommon>::V;

    /// Eqn is required for solvers (with static lifetime for pyoil3).
    pub type Eqn<'a> = diffsl::DiffSl<'a, M>;
}
