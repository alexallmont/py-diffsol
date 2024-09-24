#[macro_export]
macro_rules! solver_class {
    (
        $PyApiName:expr,
        $RustType:tt,
        $InterfaceHandle:tt,
        $ConstructorFn:tt
    ) => {
        pyoil3_class!($PyApiName, $RustType, $InterfaceHandle);

        #[pymethods]
        impl $InterfaceHandle::PyClass {
            #[new]
            pub fn new() -> $InterfaceHandle::PyClass {
                $InterfaceHandle::PyClass::bind_instance(
                    RefCell::new($ConstructorFn())
                )
            }

            pub fn step<'py>(
                slf: PyRefMut<'py, Self>
            ) -> PyResult<SolverStopReason> {
                let solver_guard = slf.0.lock().unwrap();
                let mut solver = solver_guard.instance.borrow_mut();
                let result = solver.step();
                let state = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
                Ok(SolverStopReason::from(state))
            }

            pub fn solve<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &pyoil_problem::PyClass,
                final_time: Option<T>
            ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
                let solver_guard = slf.0.lock().unwrap();
                let mut solver = solver_guard.instance.borrow_mut();
                let problem_guard = problem.0.lock().unwrap();
                let final_time = final_time.unwrap_or(1.0);
                let result = solver.solve(&problem_guard.ref_static, final_time);
                let (y, t) = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
                Ok((
                    vec_v_to_pyarray::<DM>(slf.py(), &y),
                    vec_t_to_pyarray(slf.py(), &t),
                ))
            }

            fn solve_dense<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &pyoil_problem::PyClass,
                t_eval: Vec<T>
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                let solver_guard = slf.0.lock().unwrap();
                let mut solver = solver_guard.instance.borrow_mut();
                let problem_guard = problem.0.lock().unwrap();
                let result = solver.solve_dense(&problem_guard.ref_static, &t_eval);
                let values = result.map_err(|err| PyValueError::new_err(err.to_string()))?;
                let pyarray = vec_v_to_pyarray::<DM>(slf.py(), &values);
                Ok(pyarray)
            }
        }
    };
}
