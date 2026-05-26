use thiserror::Error;

#[derive(Debug, Error)]
pub enum MathError {
    #[error("Could not calculate the CRB!")]
    CrbError,

    #[error("Could not invert the FIM")]
    InvError(#[from] ndarray_linalg::error::LinalgError),

    #[error("OpenCl error: {0}")]
    OpenClError(#[from] opencl3::error_codes::ClError),

    #[error("NdArrayShape error: {0}")]
    NdArrayShapeError(#[from] ndarray::ShapeError),
}
