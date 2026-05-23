use std::f64::consts::PI;

use ndarray::{Zip, prelude::*};
use num_complex::Complex;
use rayon::prelude::*;

use crate::tm::TM;

/// Calculate beam responses given each sources lmn
pub(crate) fn calc_jones_response(tm: &TM, source_lmn: &Array2<f64>, lambda_m: f64) -> Array1<f64> {
    let station_layout = &tm.station_layout;
    let num_elems = station_layout.len();

    // Iterate over rows (directions) in parallel
    let responses: Array1<f64> = Array1::from_vec(
        source_lmn
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                // Calculate array factor
                let array_factor: Complex<f64> = (0..num_elems)
                    .into_iter()
                    .map(|i| {
                        let x_loc = station_layout[i].x;
                        let y_loc = station_layout[i].y;

                        let tot_phase = (-x_loc * row[0] + y_loc * row[1]) / lambda_m;
                        let angle = -2.0 * PI * tot_phase;

                        Complex::from_polar(1.0, angle)
                    })
                    .sum();

                let af_norm = array_factor / num_elems as f64;

                af_norm.norm_sqr()
            })
            .collect(),
    );

    return responses;
}

pub(crate) fn attenuate_with_beam(source_intensities: &mut Array1<f64>, beam: &Array1<f64>) {
    *source_intensities *= beam;
}
