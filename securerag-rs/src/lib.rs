pub mod builders;
pub mod core;
pub mod dp;
pub mod engines;
pub mod protocol;
pub mod types;

#[cfg(feature = "python-bridge")]
pub mod pyo3_bridge;

#[cfg(feature = "python-bridge")]
use pyo3::prelude::*;

#[cfg(feature = "python-bridge")]
use crate::pyo3_bridge::BackendBridge;

#[cfg(feature = "python-bridge")]
#[pymodule]
fn securerag_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BackendBridge>()?;
    Ok(())
}
