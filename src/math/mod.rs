pub mod cpu;
pub mod error;

use error::MathError;
use num_complex::*;
use std::f64::consts::PI;

use crate::tm;
use crate::tm::TM;
use ndarray::prelude::*;
use ndarray_linalg::*;
use rayon::prelude::*;

/// Check out: https://www.cv.nrao.edu/~sransom/web/Ch3.html
pub(crate) fn calc_re(
    tm: &TM,
    t_sys: f64,
    channel_width: f64,
    int_time: f64,
    tel_name: &String, // SHould I do something with this? (MWA)
) -> (f64, f64) {
    let n_stations = tm.get_num_station() as f64;

    let r = tm.station_diam;

    // NOTE: We are assuming 100% collecting area, and that the area is circular
    let a_eff: f64 = PI * (r / 2.0).powi(2);
    // let a_eff = (r / 2.0).powi(2);

    let k = physical_constants::BOLTZMANN_CONSTANT;

    // "Single dish" sensitivity. A station in the MWA/SKA are phased arrays that "act" like single
    // dishes (can be steered).
    let radiometer =
        10.0f64.powi(26) * (2.0 * k * t_sys) / (a_eff * (channel_width * int_time).sqrt());

    let radio_interferometer = 5.0 * radiometer / (n_stations * (n_stations - 1.0)).sqrt();

    (radiometer, radio_interferometer)
}

pub(crate) fn create_baselines(tel_layout: &tm::CoordinateList) -> Array3<f64> {
    let n_tiles = tel_layout.len();
    let mut baselines_xy = Array3::<f64>::zeros((n_tiles, n_tiles, 2));

    for (i, ant_i) in tel_layout.iter().enumerate() {
        for (j, ant_j) in tel_layout.iter().enumerate() {
            baselines_xy[[i, j, 0]] = ant_i.x - ant_j.x;
            baselines_xy[[i, j, 1]] = ant_i.y - ant_j.y;
        }
    }

    return baselines_xy;
}

pub(crate) trait Executor {
    fn calculate_crb(
        &self,
        n_ant: usize,
        baselines_xy: &Array3<f64>,
        source_intensities: &Array1<f64>,
        source_lmn: &Array2<f64>,
        lambda: f64,
        sigma: f64,
    ) -> Result<Array2<f64>, MathError>;
}
