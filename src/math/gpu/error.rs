use thiserror::Error;

#[derive(Debug, Error)]
pub enum GpuError {
    #[error("OpenCl error: {0}")]
    OpenClError(#[from] opencl3::error_codes::ClError),
}
