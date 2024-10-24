//! Nalgebra to Python conversion methods

use pyo3::{prelude::*, types::PyList};
use diffsol::matrix::MatrixCommon;
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};

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

pub fn set_v_from_py<'py>(vec_v: &mut DVector<T>, py_array: &PyReadonlyArray1<'py, T>) -> PyResult<()> {
    let slice = py_array.as_slice()?;
    *vec_v = DVector::from_column_slice(slice);
    Ok(())
}

pub fn set_vec_v_from_py<'py>(vec_v: &mut Vec<V>, py_list: Bound<'py, PyList>) -> PyResult<()> {
    // Map all elements out of the list first and use downcast check they are all
    // valid numpy arrays (any errors will bail out early). collect automatically
    // extracts the OK value from the result.
    *vec_v = py_list.iter().map(|py_item| {
        let py_array = py_item
            .downcast::<PyArray1<T>>()?
            .readonly();
        let slice = py_array.as_slice()?;
        Ok(DVector::from_column_slice(slice))
    }).collect::<Result<Vec<_>, PyErr>>()?;
    Ok(())
}
