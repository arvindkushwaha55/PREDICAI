//! PREDICAI Rust edge engine library (Phase D, plan.md).
//! The binary in `main.rs` is a thin bootstrap over this library so
//! integration tests can build the real router.

pub mod maintenance;
pub mod model;
pub mod preprocess;
pub mod routes;
pub mod types;
