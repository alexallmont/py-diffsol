//! Nalgebra to Python conversion methods

use pyo3::{Bound, Python, types::PyList};
use numpy::PyArray1;

type M = nalgebra::DMatrix<f64>;
type T = <M as diffsol::matrix::MatrixCommon>::T;
type V = <M as diffsol::matrix::MatrixCommon>::V;

pub fn v_to_py<'py>(v: &V, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_slice_bound(py, v.as_slice())
}

pub fn vec_t_to_py<'py>(vec_t: &Vec<T>, py: Python<'py>) -> Bound<'py, PyArray1<T>> {
    PyArray1::from_slice_bound(py, vec_t.as_slice())
}

pub fn vec_v_to_py<'py>(vec_v: &Vec<V>, py: Python<'py>) -> Bound<'py, PyList> {
    PyList::new_bound(py, vec_v.iter().map(|v| {
        PyArray1::from_slice_bound(py, v.as_slice())
    }))
}
