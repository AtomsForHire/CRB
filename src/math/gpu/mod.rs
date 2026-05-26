use super::error::MathError;
use num_complex::*;
use std::f64::consts::PI;

use super::Executor;
use crate::tm;
use crate::tm::TM;
use ndarray::prelude::*;
use ndarray_linalg::*;
use rayon::prelude::*;

pub(crate) struct GpuExecutor;

impl GpuExecutor {
    pub(crate) fn new() -> Self {
        Self
    }
}

impl Executor for GpuExecutor {
    fn calculate_crb(
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
    }
}
