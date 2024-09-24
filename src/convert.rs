use pyo3::{
    prelude::*,
    Bound,
};
use numpy::{
    PyArray1,
    PyArray2,
    PyArrayMethods,
    ToPyArray
};
use diffsol::vector::Vector;

/// Convert Vec<Vec<V>> to 2D ndarray
///
/// The number of columns in the output array is the minimum sized vec inside
/// the main vec. This is a safeguard whilst diffsol uses this type to avoid
/// any chance of a buffer overrun.
pub fn vec_v_to_pyarray<'py, M: diffsol::matrix::DenseMatrix>(
    py: Python<'py>,
    vec: &Vec<M::V>
) -> Bound<'py, PyArray2<f64>> {
    let nrows = vec.len();
    let ncols = vec.iter().map(|v| v.len()).min().unwrap_or(0);
    let arr = unsafe {
        PyArray2::<f64>::new_bound(
            py,
            [nrows, ncols],
            false
        )
    };

    for r in 0..nrows {
        for c in 0..ncols {
            unsafe {
                let sub_elem = vec[r][c];
                arr.uget_raw([r, c]).write(sub_elem.into());
            }
        }
    }

    arr
}

pub fn vec_t_to_pyarray<'py>(
    py: Python<'py>,
    vec: &Vec<f64>
) -> Bound<'py, PyArray1<f64>> {
    let pyarray = vec.to_pyarray_bound(py);
    pyarray
}
