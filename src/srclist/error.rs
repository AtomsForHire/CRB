// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use thiserror::Error;

// use crate::{io::GlobError, srclist::HYPERDRIVE_SOURCE_LIST_FILE_TYPES_COMMA_SEPARATED};

/// Errors associated with reading in any kind of source list.
#[derive(Error, Debug)]
pub(crate) enum ReadSourceListError {
    #[error(
        "Source list error: Attempted to use RA {0}°, but this is out of range (0° <= RA < 360°)"
    )]
    InvalidRa(f64),

    #[error("Source list error: Attempted to use HA {0}, but this is out of range (0 < HA < 24)")]
    InvalidHa(f64),

    #[error(
        "Source list error: Attempted to use Dec {0}°, but this is out of range (-90° <= Dec <= 90°)"
    )]
    InvalidDec(f64),

    #[error("Source {source_name}: The sum of all Stokes {stokes_comp} flux densities was negative ({sum})")]
    InvalidFluxDensitySum {
        sum: f64,
        stokes_comp: &'static str,
        source_name: String,
    },

    #[error("Source {source_name}: A component contains NaNs for its flux densities. This is not allowed.")]
    NaNsInComponent { source_name: String },

    #[error("Could not interpret the contents of the source list. Specify which style source list it is, and a more specific error can be shown.")]
    FailedToReadAsAnyType,

    #[error("Could not deserialise the contents as yaml or json.\n\nyaml error: {yaml_err}\n\njson error: {json_err}")]
    FailedToDeserialise { yaml_err: String, json_err: String },

    #[error("No sky-model source list file supplied")]
    NoSourceList,

    // #[error(transparent)]
    // Glob(#[from] GlobError),
    #[error("The number of specified sources was 0, or the size of the source list was 0")]
    NoSources,

    #[error("After vetoing sources, none were left. Decrease the veto threshold, or supply more sources")]
    NoSourcesAfterVeto,

    #[error("Tried to use {requested} sources, but only {available} sources were available after vetoing")]
    VetoTooFewSources { requested: usize, available: usize },

    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),

    #[error(transparent)]
    Sexagesimal(#[from] marlu::sexagesimal::SexagesimalError),

    #[error(transparent)]
    IO(#[from] std::io::Error),
}
