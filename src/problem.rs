use diffsol::OdeSolverProblem;

use pyoil3::pyoil3_ref_class;
use crate::context::pyoil_context;

// Type is aliased because pyoil3 cannot yet parse generics
pub type SolverProblem<'a> = OdeSolverProblem<diffsol::ode_solver::diffsl::DiffSl<'a>>;

pyoil3_ref_class!(
    "OdeSolverProblem",
    SolverProblem,
    pyoil_problem,
    pyoil_context
);
