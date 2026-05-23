pub(crate) mod error;
pub(crate) mod read;
pub(crate) mod types;

use itertools::Itertools;
use strum::IntoEnumIterator;
pub use types::*;

/// All of the possible file extensions that a hyperdrive-style sky-model source
/// list can have.
#[derive(
    Debug, Clone, Copy, strum_macros::Display, strum_macros::EnumIter, strum_macros::EnumString,
)]
pub(crate) enum HyperdriveFileType {
    #[strum(serialize = "yaml")]
    Yaml,

    #[strum(serialize = "json")]
    Json,
}

lazy_static::lazy_static! {
    pub(crate) static ref HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED: String = HyperdriveFileType::iter().join(", ");
}
