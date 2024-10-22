//! Nalgebra to Python conversion methods

use pyo3::{prelude::*, exceptions::PyValueError, types::PyList};
use diffsol::matrix::MatrixCommon;
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1};

type M = DMatrix<f64>;
type T = <M as MatrixCommon>::T;
type V = <M as MatrixCommon>::V;

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

pub fn set_v_from_py<'py>(vec_v: &mut DVector<T>, py_value: &PyReadonlyArray1<'py, T>) -> PyResult<()> {
    let slice = py_value.as_slice().map_err(|err| PyValueError::new_err(err))?;
    *vec_v = DVector::from_column_slice(slice);
    Ok(())
}

