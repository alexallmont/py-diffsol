use numpy::{
    PyArray2,
    PyArrayMethods
};

use crate::core::types;
use pyo3::prelude::*;

/// Convert the results of a matrix
pub fn matrix_to_py<'py>(
    py: Python<'py>,
    matrix: &types::SolveMatrixType
) -> pyo3::Bound<'py, PyArray2<f64>> {
    // FIXME improve init operation
    let r = matrix.nrows();
    let c = matrix.ncols();
    let result = unsafe {
        let arr = PyArray2::<f64>::new_bound(py, [r, c], false);
        for i in 0..r {
            for j in 0..c {
                arr.uget_raw([i, j]).write(*matrix.index((i, j)));
            }
        }
        arr
    };

    result
}
