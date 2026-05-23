pub mod error;
use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

use crate::config::Config;

pub struct TM {
    pub(crate) array_layout: CoordinateList,
    pub(crate) station_layout: CoordinateList, // TODO: Assume same stations throughout array for now
    pub(crate) station_diam: f64,
}

impl TM {
    pub(crate) fn new(config: &Config) -> Result<Self, error::TMError> {
        let array_layout = read_coordinates_from_file(&config.tel_config.telescope_layout_file)?;
        let station_layout = read_coordinates_from_file(&config.tel_config.station_layout_file)?;

        Ok(Self {
            array_layout,
            station_layout,
            station_diam: config.tel_config.station_diameter,
        })
    }

    pub(crate) fn get_num_station(&self) -> usize {
        return self.array_layout.len();
    }
}

pub type CoordinateList = Vec<Coordinate>;

pub struct Coordinate {
    pub x: f64,
    pub y: f64,
}

fn read_coordinates_from_file(filename: &Path) -> Result<CoordinateList, error::TMError> {
    let file = fs::File::open(filename)?; // Open the file, returns Result
    let reader = io::BufReader::new(file);
    let mut coordinates = Vec::new();
    let mut line_number = 0;

    for line_result in reader.lines() {
        line_number += 1;
        let line = line_result?; // Propagate IO errors

        let parts: Vec<&str> = line.trim().split(',').collect();

        if parts.len() == 2 {
            if let (Ok(x), Ok(y)) = (parts[0].trim().parse(), parts[1].trim().parse()) {
                coordinates.push(Coordinate { x, y });
            } else {
                eprintln!(
                    "Warning: Invalid coordinate format on line {}: '{}'. Skipping line.",
                    line_number, line
                );
            }
        } else if !line.trim().is_empty() {
            eprintln!(
                "Warning: Malformed line (expected two comma-separated values) on line {}: '{}'. Skipping line.",
                line_number, line
            );
        }
        // Empty lines are implicitly skipped by the loop
    }

    Ok(coordinates)
}
