// FIXME rename this file. Solution type is no longer in diffsol

use numpy::{PyArray1, PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{prelude::*, Bound};
use crate::core::types::{T, V};

/// Public wrappers for 1D and 2D numpy arrays
pub type BoundPyArray1<'py> = Bound<'py, PyArray1<T>>;
pub type BoundPyArray2<'py> = Bound<'py, PyArray2<T>>;

/// Convert Vec<Vec<V>> to 2D ndarray
///
/// The number of columns in the output array is the minimum sized vec inside
/// the main vec. This is a safeguard whilst diffsol uses this type to avoid
/// any chance of a buffer overrun.
pub fn vec_v_to_pyarray<'py>(
    py: Python<'py>,
    vec: &Vec<V>
) -> BoundPyArray2<'py> {
    let nrows = vec.len();
    let ncols = vec.iter().map(|v| v.len()).min().unwrap_or(0);
    let arr = unsafe {
        PyArray2::<T>::new_bound(
            py,
            [nrows, ncols],
            false
        )
    };

    for r in 0..nrows {
        for c in 0..ncols {
            unsafe {
                let elem = vec.get_unchecked(r);
                let sub_elem = elem.get_unchecked(c);
                arr.uget_raw([r, c]).write(*sub_elem);
            }
        }
    }

    arr
}

pub fn vec_t_to_pyarray<'py>(
    py: Python<'py>,
    vec: &Vec<T>
) -> BoundPyArray1<'py> {
    let pyarray = vec.to_pyarray_bound(py);
    pyarray
}

#[cfg(test)]
mod tests {
    use numpy::ndarray::{arr2, Array2};
    use super::*;

    #[test]
    fn test_vec_v_to_pyarray() {
        pyo3::Python::with_gil(|py| {
            let mut rows = Vec::<V>::new();

            // Initially test empty case
            {
                let pyarray = vec_v_to_pyarray(py, &rows);
                assert_eq!(pyarray.len().unwrap(), 0);
                assert_eq!(pyarray.readonly().as_array(), Array2::zeros([0, 0]));
            }

            // Check that a 3x1 addition ends up as a 1 new row of 3 columns.
            {
                rows.push(V::from_vec(vec!(1.0, 2.0, 3.0)));
                let pyarray = vec_v_to_pyarray(py, &rows);
                assert_eq!(pyarray.len().unwrap(), 1);
                assert_eq!(pyarray.readonly().as_array(), arr2(&[[1.0, 2.0, 3.0]]));
            }

            // Edge case of adding 2x1 afterwards results in 2 rows of 2 columns
            // because the conversion avoids overrunning the minimum col length.
            {
                rows.push(V::from_vec(vec!(4.0, 5.0)));
                let pyarray = vec_v_to_pyarray(py, &rows);
                assert_eq!(pyarray.len().unwrap(), 2);
                assert_eq!(pyarray.readonly().as_array(), arr2(&[[1.0, 2.0], [4.0, 5.0]]));
            }
        });
    }
}
