use thiserror::Error;

#[derive(Debug, Error)]
pub enum MathError {
    #[error("Could not calculate the CRB!")]
    CrbError,

    #[error("Could not invert the FIM")]
    InvError(#[from] ndarray_linalg::error::LinalgError),
}
