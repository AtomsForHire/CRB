pub mod error;

use super::error::MathError;
use num_complex::*;
use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::program::Program;
use opencl3::types::{CL_BLOCKING, cl_double, cl_uint};
use std::f64::consts::PI;
use std::ptr;

use super::Executor;
use crate::tm;
use crate::tm::TM;
use ndarray::prelude::*;
use ndarray_linalg::*;
use opencl3::command_queue::{CL_QUEUE_PROFILING_ENABLE, CommandQueue};
use opencl3::context::Context;
use opencl3::device::{CL_DEVICE_TYPE_GPU, Device, get_all_devices};
use opencl3::kernel::{ExecuteKernel, Kernel};
use rayon::prelude::*;

pub(crate) struct GpuExecutor {
    context: Context,
    command_queue: CommandQueue,
    kernel: Kernel,
}

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
        let program = Program::create_and_build_from_source(&context, KERNEL, "")
            .expect("Could not create program");

        // Create kernel
        let kernel = Kernel::create(&program, "calculate_crb_kernel")?;

        Ok(Self {
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
        // Mostly followed the example on the opencl3 crate: https://github.com/kenba/opencl3/blob/main/examples/basic.rs
        let num_sources: usize = source_intensities.len();

        // Flatten baselines
        let baselines_x: Vec<f64> = baselines_xy
            .slice(s![.., .., 0])
            .flatten_with_order(ndarray::Order::RowMajor)
            .into_iter()
            .collect();

        let baselines_y: Vec<f64> = baselines_xy
            .slice(s![.., .., 1])
            .flatten_with_order(ndarray::Order::RowMajor)
            .into_iter()
            .collect();

        let n_d_baselines = baselines_x.len();

        let source_l: Vec<f64> = source_lmn.slice(s![.., 0]).iter().copied().collect();
        let source_m: Vec<f64> = source_lmn.slice(s![.., 1]).iter().copied().collect();
        let source_intensities_vec: Vec<f64> = source_intensities.iter().copied().collect();

        // Create device buffers
        let mut d_baselines_x = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                n_d_baselines,
                ptr::null_mut(),
            )?
        };
        let mut d_baselines_y = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                n_d_baselines,
                ptr::null_mut(),
            )?
        };
        let mut d_source_intensities = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                num_sources,
                ptr::null_mut(),
            )?
        };
        let mut d_source_l = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                num_sources,
                ptr::null_mut(),
            )?
        };
        let mut d_source_m = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_READ_ONLY,
                num_sources,
                ptr::null_mut(),
            )?
        };
        let mut d_results = unsafe {
            Buffer::<cl_double>::create(
                &self.context,
                CL_MEM_WRITE_ONLY,
                n_ant * n_ant,
                ptr::null_mut(),
            )?
        };

        // Write data to buffers
        let _write_baselines_x = unsafe {
            &self.command_queue.enqueue_write_buffer(
                &mut d_baselines_x,
                CL_BLOCKING,
                0,
                &baselines_x,
                &[],
            )?
        };
        let _write_baselines_y = unsafe {
            &self.command_queue.enqueue_write_buffer(
                &mut d_baselines_y,
                CL_BLOCKING,
                0,
                &baselines_y,
                &[],
            )?
        };
        let _write_sources_intensities = unsafe {
            &self.command_queue.enqueue_write_buffer(
                &mut d_source_intensities,
                CL_BLOCKING,
                0,
                &source_intensities_vec,
                &[],
            )?
        };
        let _write_sources_l = unsafe {
            &self.command_queue.enqueue_write_buffer(
                &mut d_source_l,
                CL_BLOCKING,
                0,
                &source_l,
                &[],
            )?
        };
        let _write_sources_m = unsafe {
            &self.command_queue.enqueue_write_buffer(
                &mut d_source_m,
                CL_BLOCKING,
                0,
                &source_m,
                &[],
            )?
        };

        let kernel_event = unsafe {
            ExecuteKernel::new(&self.kernel)
                .set_arg(&(n_ant as i32))
                .set_arg(&d_baselines_x)
                .set_arg(&d_baselines_y)
                .set_arg(&(num_sources as i32))
                .set_arg(&d_source_intensities)
                .set_arg(&d_source_l)
                .set_arg(&d_source_m)
                .set_arg(&lambda)
                .set_arg(&sigma)
                .set_arg(&d_results)
                .set_global_work_size(n_ant * n_ant)
                .enqueue_nd_range(&self.command_queue)?
        };

        // Read back results
        let mut results = vec![0.0f64; n_ant * n_ant];
        unsafe {
            self.command_queue.enqueue_read_buffer(
                &d_results,
                CL_BLOCKING,
                0,
                &mut results,
                &[kernel_event.get()], // wait for kernel to finish
            )?
        };

        let fim = Array2::from_shape_vec((n_ant, n_ant), results)?;
        let crb = fim.inv()?;
        Ok(crb)
    }
}
