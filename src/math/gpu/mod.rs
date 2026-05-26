pub mod error;

use super::error::MathError;
use num_complex::*;
use opencl3::program::Program;
use std::f64::consts::PI;

use super::Executor;
use crate::tm;
use crate::tm::TM;
use ndarray::prelude::*;
use ndarray_linalg::*;
use rayon::prelude::*;
use opencl3::context::Context;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::kernel::Kernel;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};

pub(crate) struct GpuExecutor{
    context: Context,
    command_queue: CommandQueue,
    kernel: Kernel,
};

const KERNEL: &str = include_str!("./crb_kernel.cl");

impl GpuExecutor {
    pub(crate) fn new() -> Result<Self, error::GpuError> {
        // Query all available devices
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU)?; 

        // Get id of first device
        let d_id = devices.first().expect("No GPU device found");

        let device = Device::new(*d_id);

        // Create context
        let context = Context::from_device(&device)?;

        // Create queue
        let command_queue = CommandQueue::create_default(&context, CL_QUEUE_PROFILING_ENABLE)?;

        // Create program
        let program = Program::create_and_build_from_source(&context, KERNEL, "").expect("Could not create program");

        // Create kernel
        let kernel = Kernel::create(&program, "CRB")?;

        Ok(Self{
            context,
            command_queue,
            kernel,
        })
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

        todo!();
    }
}
