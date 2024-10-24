//! Python wrapper for diffsol/src/ode_solver/method.rs
//! Implemented as macro as this is duplicated for all solver types, and having
//! a macro generate a separate PyClass for each - instead of using a strategy
//! pattern, for example - makes better use of PyO3's type system.
//! Note that problem, take_state and checkpoint methods are ommitted by design,
//! and state_mut is ommitted because in Python all state access is routed via
//! py_solver_state, e.g. get with s = solver.state, set with solver.state = s.

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

            // TODO fn set_problem(&mut self, state: OdeSolverState<Eqn::V>, problem: &OdeSolverProblem<Eqn>);

            pub fn step<'py>(
                slf: PyRefMut<'py, Self>
            ) -> PyResult<SolverStopReason> {
                slf.lock(|solver| {
                    let state = solver.borrow_mut().step().map_err(diffsol_err)?;
                    Ok(SolverStopReason::from(state))
                })
            }

            // TODO fn set_stop_time(&mut self, tstop: Eqn::T) -> Result<(), DiffsolError>;
            // TODO fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError>;
            // TODO fn interpolate_sens(&self, t: Eqn::T) -> Result<Vec<Eqn::V>, DiffsolError>;

            #[getter]
            fn state<'py>(slf: PyRefMut<'py, Self>) -> py_solver_state::PyClass {
                // State is accessed as direct reference from this solver
                py_solver_state::PyClass::new_binding(
                    // Note that $RustType is used here to select SolverState::Bdf
                    // or SolverState::Sdirk enum depending on solver type so the
                    // state can be retrieved from the Arc<Mutex<solver>> later.
                    SolverState::$RustType(slf.0.clone())
                )
            }

            pub fn order<'py>(slf: PyRefMut<'py, Self>) -> u64 {
                slf.lock(|solver| { solver.borrow().order() }) as u64
            }

            #[pyo3(signature = (problem, final_time=1.0))]
            pub fn solve<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &py_problem::PyClass,
                final_time: T
            ) -> PyResult<(Bound<'py, PyList>, Bound<'py, PyArray1<T>>)> {
                slf.lock(|solver| {
                    problem.lock(|prb| {
                        let (y, t) = solver.borrow_mut().solve(
                            prb,
                            final_time
                        ).map_err(diffsol_err)?;

                        Ok((
                            py_convert::vec_v_to_py(&y, slf.py()),
                            py_convert::vec_t_to_py(&t, slf.py())
                        ))
                    })
                })
            }

            fn solve_dense<'py>(
                slf: PyRefMut<'py, Self>,
                problem: &py_problem::PyClass,
                t_eval: Vec<T>
            ) -> PyResult<Bound<'py, PyList>> {
                slf.lock(|solver| {
                    problem.lock(|prb| {
                        let values = solver.borrow_mut().solve_dense(prb, &t_eval).map_err(diffsol_err)?;
                        Ok(py_convert::vec_v_to_py(&values, slf.py()))
                    })
                })
            }
        }
    };
}
