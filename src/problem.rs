use diffsol::OdeSolverProblem;
use pyoil3::pyoil3_ref_class;

use crate::{
    context::pyoil_context,
    core::types
};

// Type is aliased until pyoil3 can parse generics
pub type SolverProblem<'a> = OdeSolverProblem<types::Eqn<'a>>;

pyoil3_ref_class!(
    "OdeSolverProblem",
    SolverProblem,
    pyoil_problem,
    pyoil_context // FIXME compile time enforement
);
