/// Macro to wrap creation of modules from bindings.rs
///
/// Each binding created requires a module name, a matrix and solver type to
/// expose the underlying diffsol classes to python.
///
/// This approach could be considered compiler abuse as the implementation sets
/// the three required values and then pulls in private/bindings.rs which in
/// turn gets them from super::<name>. This is not pretty but it's the neatest
/// approach to coerce templated types into a common suite of module classes for
/// PyO3 without duplicating a lot of boilerplate and having upcasting issues.
#[macro_export]
macro_rules! create_binding {
    (
        $module_name:tt,
        $matrix_type:tt,
        $solver_type:tt,
        $py_convert:tt
    ) => {
        #[path="."]
        pub mod $module_name {
            use super::$matrix_type;
            use super::$solver_type;
            use super::$py_convert as py_convert;

            // Module name, underlying type name and type
            static MODULE_NAME:&'static str = stringify!($module_name);
            type Matrix = $matrix_type;
            type LinearSolver<Op> = $solver_type<Op>;

            // bindings accesses above values and types via super::
            mod bindings;
            pub use bindings::*;
        }
    };
}
