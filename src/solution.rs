use numpy::{
    PyArray1,
    PyArray2,
    PyArrayMethods
};
use pyo3::prelude::*;

use crate::core::types;

/// Public wrappers for 1D and 2D numpy arrays
pub type BoundPyArray1<'p> = pyo3::Bound<'p, numpy::PyArray1<types::T>>;
pub type BoundPyArray2<'p> = pyo3::Bound<'p, numpy::PyArray2<types::T>>;

/// PyO3 wrapper for a diffsol OdeSolution
#[pyclass]
#[pyo3(name = "OdeSolution")]
pub struct PyOdeSolution(pub diffsol::OdeSolution<types::V>);

#[pymethods]
impl PyOdeSolution {
    /// Get times as a 1D ndarray
    #[getter]
    pub fn t<'p>(&self, py: Python<'p>) -> BoundPyArray1<'p> {
        let len = self.0.t.len();
        let arr = unsafe {
            PyArray1::<types::T>::new_bound(
                py, [len], false
            )
        };
        unsafe {
            // FIXME set direct from ptr
            for i in 0..len {
                let elem = self.0.t.get_unchecked(i);
                arr.uget_raw([i]).write(*elem);
            }
        }
        arr
    }

    /// Get ys as a 2D ndarray
    #[getter]
    pub fn y<'p>(&self, py: Python<'p>) -> BoundPyArray2<'p> {
        match self.0.y.len() {
            0 => {
                let arr = unsafe {
                    PyArray2::<types::T>::new_bound(
                        py, [0, 0], false
                    )
                };
                arr
            },
            len => {
                // FIXME safe to assume all elements equal?
                let ys_len = self.0.y.get(0).unwrap().len();
                let arr = unsafe {
                    PyArray2::<types::T>::new_bound(
                        py, [len, ys_len], false
                    )
                };
                // FIXME set direct from ptr
                for i in 0..len {
                    for j in 0..ys_len {
                        unsafe {
                            let elem = self.0.y.get_unchecked(i);
                            let sub_elem = elem.get_unchecked(j);
                            arr.uget_raw([i, j]).write(*sub_elem);
                        }
                    }
                }
                arr
            }
        }
    }
}
