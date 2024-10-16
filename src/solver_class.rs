// Python wrapper for diffsol/src/ode_solver/method.rs

#[macro_export]
macro_rules! solver_class {
    (
        $PyApiName:expr,
        $RustType:tt,
        $InterfaceHandle:tt,
        $ConstructorFn:tt
    ) => {
        py_class!($PyApiName, $RustType, $InterfaceHandle);

        #[pymethods]
        impl $InterfaceHandle::PyClass {
            #[new]
            pub fn new() -> $InterfaceHandle::PyClass {
                $InterfaceHandle::PyClass::new_binding(
                    RefCell::new($ConstructorFn())
                )
            }

            pub fn step<'py>(
                slf: PyRefMut<'py, Self>
            ) -> PyResult<SolverStopReason> {
                slf.use_inst(|solver| {
                    let state = solver.borrow_mut().step().map_err(diffsol_err)?;
                    Ok(SolverStopReason::from(state))
                })
            }

            pub fn order<'py>(slf: PyRefMut<'py, Self>) -> u64 {
                slf.use_inst(|solver| { solver.borrow().order() }) as u64
            }

            pub fn solve<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &py_problem::PyClass,
                final_time: Option<T>
            ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<f64>>)> {
                slf.use_inst(|solver| {
                    let mut solver = solver.borrow_mut();
                    problem.use_inst(|prb| {
                        let final_time = final_time.unwrap_or(1.0);
                        let (y, t) = solver.solve(prb, final_time).map_err(diffsol_err)?;
                        Ok((
                            vec_v_to_pyarray::<DM>(slf.py(), &y),
                            vec_t_to_pyarray(slf.py(), &t),
                        ))
                    })
                })
            }

            fn solve_dense<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &py_problem::PyClass,
                t_eval: Vec<T>
            ) -> PyResult<Bound<'py, PyArray2<f64>>> {
                slf.use_inst(|solver| {
                    let mut solver = solver.borrow_mut();
                    problem.use_inst(|prb| {
                        let values = solver.solve_dense(prb, &t_eval).map_err(diffsol_err)?;
                        let pyarray = vec_v_to_pyarray::<DM>(slf.py(), &values);
                        Ok(pyarray)
                    })
                })
            }
        }
    };
}
