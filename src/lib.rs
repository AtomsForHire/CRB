use std::{
    fs, io,
    path::{Path, PathBuf},
};
mod beam;
mod math;
mod save;
use save::WriteToFile;
mod srclist;
mod tm;
use marlu::RADec;
use ndarray::prelude::*;
use srclist::*;

use log::{debug, info, warn};
use thiserror::Error;

use crate::{
    config::Config,
    math::{
        Executor,
        cpu::CpuExecutor,
        create_baselines,
        gpu::{self, GpuExecutor},
    },
};

pub mod config;

#[derive(Debug, Error)]
pub enum RunError {
    #[error("Configuration error: {0}")]
    Config(#[from] config::error::ConfigError),

    #[error("IO error: {0}")]
    IO(#[from] std::io::Error),

    #[error("Srclist error: {0}")]
    SrclistError(#[from] srclist::error::ReadSourceListError),

    #[error("TM error: {0}")]
    TmError(#[from] tm::error::TMError),

    #[error("Ndarray error: {0}")]
    NdError(#[from] ndarray::ShapeError),

    #[error("Math error: {0}")]
    MathError(#[from] math::error::MathError),

    #[error("GPU error: {0}")]
    GpuError(#[from] gpu::error::GpuError),
}

fn compare_f64(a: &f64, b: &f64) -> std::cmp::Ordering {
    match a.partial_cmp(b) {
        Some(order) => order,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater, // NaNs come after non-NaNs
            (false, true) => std::cmp::Ordering::Less,    // Non-NaNs come before NaNs
            (false, false) => unreachable!(),
        },
    }
}

pub fn run(config_path: PathBuf) -> Result<(), RunError> {
    info!("Reading in config from {:?}", &config_path);
    let config = Config::from_file(&config_path)?;

    info!("Config:\n {:#?}", &config);

    info!("Reading in srclist");

    let file = fs::File::open(&config.srclist)?;
    let mut buf_reader = io::BufReader::new(file);
    let source_list: SourceList = read::source_list_from_yaml(&mut buf_reader)?;

    let mut component_list: ComponentList = ComponentList::new(source_list);
    info!(
        "Number of components in source list: {}",
        component_list.len()
    );

    info!("Reading in telescope layout");
    let tm = tm::TM::new(&config)?;
    let n_stations = tm.get_num_station();
    info!("Number of stations in layout: {n_stations}");

    let (re, ri) = math::calc_re(
        &tm,
        config.obs_config.t_sys,
        config.obs_config.channel_width,
        config.obs_config.int_time,
        &config.tel_config.telescope,
    );

    info!("RE: {re} Jy");
    info!("RI: {ri} Jy");

    let baselines_xy = create_baselines(&tm.array_layout);

    // Perform an early veto by fov
    let phase_centre = RADec::from_degrees(config.obs_config.ra, config.obs_config.dec);
    info!("Phase center: {:#?}", phase_centre);

    component_list.veto_by_fov(
        phase_centre,
        physical_constants::SPEED_OF_LIGHT_IN_VACUUM / config.obs_config.start_freq,
        config.tel_config.station_diameter,
    );

    info!(
        "Number of components after fov veto: {}",
        component_list.len()
    );

    // Create frequency array
    let num_freq: usize = ((config.obs_config.end_freq - config.obs_config.start_freq)
        / config.obs_config.channel_width)
        .floor() as usize;
    let freq_array = Array::linspace(
        config.obs_config.start_freq,
        config.obs_config.end_freq,
        num_freq,
    );

    // Saving arrays
    let mut gain_unc_array = Array2::<f64>::zeros((n_stations, num_freq));
    let mut crb_row_per_freq = Array2::<f64>::zeros((num_freq, n_stations));

    let executor: Box<dyn Executor> = if config.use_gpu {
        Box::new(GpuExecutor::new()?)
    } else {
        Box::new(CpuExecutor::new())
    };

    // Perform the loop
    for (freq_idx, freq) in freq_array.iter().enumerate() {
        info!("=== Freq: {freq:.1} ===");
        let lambda_m = physical_constants::SPEED_OF_LIGHT_IN_VACUUM / *freq;
        let mut freq_comp_list = component_list.clone();

        // =====================================================================
        // Estimate flux density at current frequency then veto
        freq_comp_list.fd_for_freq(*freq);
        freq_comp_list.veto_by_flux(ri);
        info!(
            "Number of components after flux veto: {}",
            &freq_comp_list.len()
        );
        freq_comp_list.veto_by_fov(phase_centre, lambda_m, config.tel_config.station_diameter);
        info!(
            "Number of components after fov veto: {}",
            &freq_comp_list.len()
        );

        let n_comps = freq_comp_list.len();
        // =====================================================================
        // Grab l,m,B values then sort.
        // These are vecs of f64s because they are contiguous in memory
        // results in a massive speed up.
        // let source_intensities = freq_comp_list.fd_for_freq(*freq);
        let source_intensities = freq_comp_list.get_intensity_list();
        let source_lmn = freq_comp_list.get_lmn_list(phase_centre);

        // Sort
        let mut indices: Vec<usize> = (0..freq_comp_list.len()).collect();
        indices.sort_by(|&i, &j| compare_f64(&source_intensities[i], &source_intensities[j]));

        let mut sorted_source_intensities: Array1<f64> = Array1::from_vec(
            indices
                .iter()
                .rev()
                .map(|&i| source_intensities[i])
                .collect(),
        );

        let sorted_source_lmn_flat: Vec<f64> = indices
            .iter()
            .rev() // replaces the .reverse()
            .flat_map(|&i| [source_lmn[i][0], source_lmn[i][1]])
            .collect();

        let mut sorted_source_lmn: Array2<f64> =
            Array2::from_shape_vec((n_comps, 2), sorted_source_lmn_flat)?;

        debug!("{:^8} {:^8} {:^8} Before attenuation", "l", "m", "B");
        for i in 0usize..3 {
            debug!(
                "{:8.4} {:8.4} {:8.4e}",
                sorted_source_lmn[[i, 0]],
                sorted_source_lmn[[i, 1]],
                sorted_source_intensities[i]
            );
        }

        debug!(
            "Total summed flux density: {}",
            sorted_source_intensities.sum()
        );

        // =====================================================================
        // Calculate beam response for all sources
        let responses = beam::calc_jones_response(&tm, &sorted_source_lmn, lambda_m);
        debug!("Beam resonses for the first 3 ordered sources:");
        for i in 0usize..3 {
            debug!("{:8.4}", responses[[i]],);
        }

        beam::attenuate_with_beam(&mut sorted_source_intensities, &responses);

        debug!("{:^8} {:^8} {:^8} After attenuation", "l", "m", "B");
        for i in 0usize..3 {
            debug!(
                "{:8.4} {:8.4} {:8.4e}",
                sorted_source_lmn[[i, 0]],
                sorted_source_lmn[[i, 1]],
                sorted_source_intensities[i]
            );
        }

        debug!(
            "Total summed flux density: {}",
            sorted_source_intensities.sum()
        );

        // =====================================================================
        // Calculate CRB
        let crb = executor.calculate_crb(
            n_stations,
            &baselines_xy,
            &sorted_source_intensities,
            &sorted_source_lmn,
            lambda_m,
            re,
        )?;

        let mean_gain_unc = crb
            .clone()
            .into_diag()
            .mapv(|x| x.sqrt())
            .mean()
            .expect("Unable to calculate mean gain unc");

        crb_row_per_freq.slice_mut(s![freq_idx, ..]).assign(
            &crb.clone()
                .slice_move(s![0, ..])
                .into_iter()
                .collect::<Array1<f64>>(),
        );

        gain_unc_array.slice_mut(s![.., freq_idx]).assign(
            &crb.clone()
                .into_diag()
                .into_iter()
                .map(|x| x.sqrt())
                .collect::<Array1<f64>>(),
        );

        debug!(
            "Sqrt of CRB Diagonal: {:?}",
            crb.clone().into_diag().mapv(|x| x.sqrt())
        );
        info!("Mean gain uncertainty: {mean_gain_unc}");
    }

    // Save results
    if !Path::new(&config.output).exists() {
        fs::create_dir(&config.output)?;
    }

    crb_row_per_freq.write_to_file(&config.output.join("crb_first_row_per_freq.txt"))?;
    gain_unc_array.write_to_file(&config.output.join("gain_uncertainties.txt"))?;

    Ok(())
}
