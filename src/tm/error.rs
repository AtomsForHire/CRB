use thiserror::Error;

#[derive(Debug, Error)]
pub enum TMError {
    #[error("Failed to read coordinates from file: {0}")]
    Io(#[from] std::io::Error),
}
