pub mod error;

use std::path::{Path, PathBuf};

use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Config {
    pub srclist: PathBuf,
    pub output: PathBuf,
    pub obs_config: ObservationConfig,
    pub tel_config: TelescopeConfig,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct ObservationConfig {
    pub obs_name: String,
    pub ra: f64,
    pub dec: f64,
    pub start_freq: f64,
    pub end_freq: f64,
    pub channel_width: f64,
    pub int_time: f64,
    pub t_sys: f64,
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct TelescopeConfig {
    pub telescope: String,
    pub station_diameter: f64,
    pub telescope_layout_file: PathBuf,
    pub station_layout_file: PathBuf,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self, error::ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;

        Ok(config)
    }
}
