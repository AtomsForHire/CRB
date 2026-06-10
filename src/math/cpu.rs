use super::error::MathError;
use log::{debug, trace};
use num_complex::*;
use std::f64::consts::PI;

use super::Executor;
use crate::tm;
use crate::tm::TM;
use ndarray::prelude::*;
use ndarray_linalg::*;
use rayon::prelude::*;

pub(crate) struct CpuExecutor;

impl CpuExecutor {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Executor for CpuExecutor {
    fn calculate_fim(
        &self,
        n_ant: usize,
        baselines_xy: &Array3<f64>,
        source_intensities: &Array1<f64>,
        source_lmn: &Array2<f64>,
        lambda: f64,
        sigma: f64,
    ) -> Result<Array2<f64>, MathError> {
        let num_sources: usize = source_intensities.len();

        let baselines = baselines_xy / lambda;

        // Initialise FIM, top left quadrant holds information on gain-gain cross variances
        // bottom right quadrant holds information on phase-phase cross variances
        // everywhere else is 0.
        let mut fim = Array2::<f64>::zeros((2 * n_ant, 2 * n_ant));

        // Populate top-left quadrant off-diagonals
        let mut top_left = fim.slice_mut(s![0..n_ant, 0..n_ant]);
        top_left
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(a, mut row)| {
                for b in (a + 1)..n_ant {
                    // off-diagonals
                    let u_ab = baselines[[a, b, 0]];
                    let v_ab = baselines[[a, b, 1]];
                    let mut s_ab: Complex64 = Complex64::new(0.0, 0.0);
                    for idx_i in 0usize..num_sources {
                        let phase_arg: f64 = -2.0
                            * PI
                            * (u_ab * source_lmn[[idx_i, 0]] + v_ab * source_lmn[[idx_i, 1]]);
                        s_ab += source_intensities[[idx_i]] * (Complex64::i() * phase_arg).exp();
                    }

                    row[[b]] = 2.0 * s_ab.norm_sqr(); // |V_(a,b)|^2 + |V_(b, a)|^2
                }
            });

        // Populate bottom-right quadrant off diagonals
        let mut bottom_right = fim.slice_mut(s![n_ant.., n_ant..]);
        bottom_right
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(a, mut row)| {
                for b in (a + 1)..n_ant {
                    // Off-diagonals
                    let u_ab = baselines[[a, b, 0]];
                    let v_ab = baselines[[a, b, 1]];
                    let mut s_ab: Complex64 = Complex64::new(0.0, 0.0);
                    for idx_i in 0usize..num_sources {
                        let phase_arg: f64 = -2.0
                            * PI
                            * (u_ab * source_lmn[[idx_i, 0]] + v_ab * source_lmn[[idx_i, 1]]);
                        s_ab += source_intensities[[idx_i]] * (Complex64::i() * phase_arg).exp();
                    }

                    row[[b]] = -2.0 * s_ab.norm_sqr(); // -|V_(a, b)|^2 - |V_(b, a)|^2
                }
            });

        // Only calculated the 'top fin' so fill in the bottom fin
        for a in 0usize..(2 * n_ant) {
            for b in a..(2 * n_ant) {
                // fim[[b, a]] = fim[[a, b]].conj();
                fim[[b, a]] = fim[[a, b]];
            }
        }

        // Now calculate the diagonals, which depend on the off-diagonal terms
        let n = fim.nrows();
        let mid = n / 2;

        let gain_sums: Vec<f64> = (0..mid).map(|i| fim.row(i).iter().sum()).collect();
        let phase_sums: Vec<f64> = (mid..2 * n_ant)
            .map(|i| fim.row(i).abs().iter().sum())
            .collect();

        let (mut first_half, mut second_half) = fim.diag_mut().split_at(Axis(0), mid);

        first_half.iter_mut().enumerate().for_each(|(i, val)| {
            let u_ab = baselines[[i, i, 0]];
            let v_ab = baselines[[i, i, 1]];
            let mut s_ab: Complex64 = Complex64::new(0.0, 0.0);
            for idx_i in 0usize..num_sources {
                let phase_arg: f64 =
                    -2.0 * PI * (u_ab * source_lmn[[idx_i, 0]] + v_ab * source_lmn[[idx_i, 1]]);
                s_ab += source_intensities[[idx_i]] * (Complex64::i() * phase_arg).exp();
            }

            *val = 4.0 * s_ab.norm_sqr() + gain_sums[i];
        });

        second_half.iter_mut().enumerate().for_each(|(i, val)| {
            *val = phase_sums[i];
        });

        // Scale all elements
        fim *= 4.0 / (sigma * sigma);

        // Scale diagonals back down
        fim.diag_mut().iter_mut().for_each(|val| *val /= 2.0);

        trace!("gain sums: {:?}", gain_sums);
        trace!("phase sums: {:?}", phase_sums);
        trace!("Fim diagonal: {:?}", fim.diag());
        trace!("Top row FIM: {:?}", fim.slice(s![0, ..]));
        trace!("Bot row FIM: {:?}", fim.slice(s![-1, ..]));

        let ref_idx = n_ant - 1;
        let ref_row = n_ant + ref_idx; // its position in the full FIM

        let indices: Vec<usize> = (0..2 * n_ant).filter(|&i| i != ref_row).collect();

        let reduced_fim = fim.select(Axis(0), &indices).select(Axis(1), &indices);

        trace!("reduced fim shape: {:?}", reduced_fim.shape());
        return Ok(reduced_fim);
        array!
    }
}
