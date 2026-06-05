use super::error::MathError;
use log::debug;
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
        let diag_term_gains = 2.0 * (n_ant as f64 + 1.0);
        let diag_term_phases = 2.0 * (n_ant as f64 - 1.0);

        // Initialise FIM, top left quadrant holds information on gain-gain cross variances
        // bottom right quadrant holds information on phase-phase cross variances
        // everywhere else is 0.
        let mut fim = Array2::<f64>::zeros((2 * n_ant, 2 * n_ant));

        // Populate top left quadrant
        let mut top_left = fim.slice_mut(s![0..n_ant, 0..n_ant]);
        top_left
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(a, mut row)| {
                for b in a..n_ant {
                    let u_ab = baselines[[a, b, 0]];
                    let v_ab = baselines[[a, b, 1]];
                    let mut s_ab: Complex64 = Complex64::new(0.0, 0.0);
                    for idx_i in 0usize..num_sources {
                        let phase_arg: f64 = -2.0
                            * PI
                            * (u_ab * source_lmn[[idx_i, 0]] + v_ab * source_lmn[[idx_i, 1]]);
                        s_ab += source_intensities[[idx_i]] * (Complex64::i() * phase_arg).exp();
                    }

                    row[[b]] = s_ab.norm_sqr();
                    if a == b {
                        row[[b]] *= diag_term_gains;
                    } else {
                        row[[b]] *= 2.0;
                    }
                }
            });

        let mut bottom_right = fim.slice_mut(s![n_ant.., n_ant..]);
        bottom_right
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(a, mut row)| {
                for b in a..n_ant {
                    let u_ab = baselines[[a, b, 0]];
                    let v_ab = baselines[[a, b, 1]];
                    let mut s_ab: Complex64 = Complex64::new(0.0, 0.0);
                    for idx_i in 0usize..num_sources {
                        let phase_arg: f64 = -2.0
                            * PI
                            * (u_ab * source_lmn[[idx_i, 0]] + v_ab * source_lmn[[idx_i, 1]]);
                        s_ab += source_intensities[[idx_i]] * (Complex64::i() * phase_arg).exp();
                    }

                    row[[b]] = s_ab.norm_sqr();
                    if a == b {
                        row[[b]] *= diag_term_phases;
                    } else {
                        row[[b]] *= -2.0;
                    }
                }
            });

        // Only calculated the 'top fin' so fill in the bottom fin
        for a in 0usize..(2 * n_ant) {
            for b in a..(2 * n_ant) {
                fim[[b, a]] = fim[[a, b]].conj();
            }
        }

        fim = 2.0 / sigma.powi(2) * fim;

        debug!(
            "Average of all FIM elements: {:?}, Var: {:?}",
            fim.mean(),
            fim.var(0.0)
        );

        // let crb = fim.inv()?;
        return Ok(fim);
    }
}
