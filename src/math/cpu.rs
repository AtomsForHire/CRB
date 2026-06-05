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
        let diag_term = 2.0 * (n_ant as f64 + 1.0);

        // Let each thread work on a different row of the matrix
        // then combine everything with 'reduce_with'
        let mut fim = (0usize..n_ant)
            .into_par_iter()
            .map(|a| {
                let mut local_fim = Array2::<f64>::zeros((n_ant, n_ant));
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

                    local_fim[[a, b]] = s_ab.norm_sqr();

                    if a == b {
                        local_fim[[a, b]] *= diag_term;
                    } else {
                        local_fim[[a, b]] *= 2.0;
                    }
                }

                local_fim
            })
            .reduce_with(|accum, local_fim| accum + local_fim)
            .ok_or_else(|| MathError::CrbError)?;

        // Only calculated the 'top fin' so fill in the bottom fin
        for a in 0usize..n_ant {
            for b in a..n_ant {
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
