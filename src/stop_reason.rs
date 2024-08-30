use diffsol::OdeSolverStopReason;
use pyo3::prelude::*;
use crate::core::types;

#[pyclass(name = "OdeSolverStopReason")]
pub enum PyOdeSolverStopReason {
    InternalTimestep { },
    RootFound { root: types::T },
    TstopReached { },
}

type OdeSolverStopReasonT = OdeSolverStopReason<types::T>;
impl From<OdeSolverStopReasonT> for PyOdeSolverStopReason {
    fn from(value: OdeSolverStopReasonT) -> Self {
        match value {
            OdeSolverStopReasonT::InternalTimestep => PyOdeSolverStopReason::InternalTimestep { },
            OdeSolverStopReasonT::RootFound(root) => PyOdeSolverStopReason::RootFound { root },
            OdeSolverStopReasonT::TstopReached => PyOdeSolverStopReason::TstopReached { },
        }
    }
}
