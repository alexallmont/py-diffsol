//! Rust lifetimes for PyO3
//!
//! A common problem for PyO3 is that it can't expose native rust classes that
//! have lifetimes because in a garbage-collected system there is no guarantee
//! on the destruction order of objects.
//! 
//! This is resolved by using stripping the object lifetimes at compile time
//! and ensuring that they are valid at runtime by wrapping all types in an
//! Arc Mutex to ensure no two threads access the data at the same time.
//! 
//! Classes that have no lifetimes are registered with the `py_class!`,
//! given the public Python API name for the class, the type of the Rust class
//! it needs to store, and an 'interface handle' which is the internal type
//! (actually a module under the hood) that wraps up the internals in Rust.
//! 
//! Register lifetime classes with `py_class_dependant!`. This takes a fourth
//! argument of the instance handle of the class that 'owns' this lifetime, i.e.
//! the one that this new class depends on. The type names are passed without
//! their lifetime parameters. Complex generic types may need type aliases so
//! the macro implementation can handle the `tt` type, which does not allow
//! generic parameters.
//! 
//! The implementation works by the `ref_class` cloning the Arc of the owning
//! object. The clone increases the reference count on the owning object so it
//! cannot be destroyed until the ref_class has released it; effectively, it is
//! using Arc shared pointers to build a dependency tree of reference counts.

/// Declare a class that has no lifetime dependencies, but which may be
/// depended upon by ref_classes.
#[macro_export]
macro_rules! py_class {
    (
        $PyApiName:expr,
        $RustType:tt,
        $InterfaceHandle:tt
    ) => {
        pub mod $InterfaceHandle {
            use std::sync::{Arc, Mutex};
            use pyo3::prelude::*;

            pub struct RustInstance {
                pub instance: super::$RustType,
            }
            pub type ArcHandle = Arc<Mutex<RustInstance>>;

            #[pyclass(unsendable)]
            #[pyo3(name = $PyApiName)]
            pub struct PyClass(pub ArcHandle);

            impl PyClass {
                pub fn new_binding(instance: super::$RustType) -> PyClass
                {
                    let inst = super::$InterfaceHandle::RustInstance {
                        instance
                    };

                    PyClass(Arc::new(Mutex::new(inst)))
                }

                pub fn lock<UseFn, UseFnReturn>(&self, use_fn: UseFn) -> UseFnReturn
                where
                    UseFn: FnOnce(&super::$RustType) -> UseFnReturn,
                {
                    let guard = self.0.lock().unwrap();
                    use_fn(&guard.instance)
                }
            }
        }
    };
}

/// Declare a ref_class that may depend upon a PyOil3 owner.
#[macro_export]
macro_rules! py_class_dependant {
    (
        $PyApiName:expr,
        $RustType:tt,
        $InterfaceHandle:tt,
        $OwnerHandle:tt
    ) => {
        pub mod $InterfaceHandle {
            use std::sync::{Arc, Mutex};
            use pyo3::prelude::*;

            pub type DependsOnType = Arc<Mutex<super::$OwnerHandle::RustInstance>>;
            pub struct RustInstance {
                pub instance: super::$RustType<'static>,
                pub depends_on: DependsOnType,
            }

            #[pyclass(unsendable)]
            #[pyo3(name = $PyApiName)]
            pub struct PyClass(pub Arc<Mutex<RustInstance>>);

            impl PyClass {
                pub fn new_binding(
                    instance: super::$RustType,
                    depends_on: DependsOnType
                ) -> PyClass {
                    // To circumvent lifetime errors the dependant class is
                    // passed around as a static instance. The lifetime is
                    // instead guaranteed by `depends_on` existing long enough
                    // by reference count.
                    let instance: super::$RustType<'static> = unsafe {
                        std::mem::transmute(instance)
                    };

                    let inst = super::$InterfaceHandle::RustInstance {
                        instance,
                        depends_on
                    };

                    PyClass(Arc::new(Mutex::new(inst)))
                }

                pub fn lock<UseFn, UseFnReturn>(&self, use_fn: UseFn) -> UseFnReturn
                where
                    UseFn: FnOnce(&super::$RustType<'static>) -> UseFnReturn,
                {
                    let guard = self.0.lock().unwrap();
                    use_fn(&guard.instance)
                }
            }
        }
    };
}