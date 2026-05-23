// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Struct to store flux densities from yaml which look like:
//! fd:
//!     freq: ...
//!     i: ...
//!     q: ...
//!     u: ...
//!
pub(crate) const DEFAULT_SPEC_INDEX: f64 = -0.8;
use serde::{Deserialize, Serialize};
// vec1 ensures there is at least one item in the vector
use vec1::Vec1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct FluxDensity {
    pub freq: f64,

    pub i: f64,

    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub q: f64,

    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub u: f64,

    #[serde(default)]
    #[serde(skip_serializing_if = "is_zero")]
    pub v: f64,
}

impl FluxDensity {
    /// Given two flux densities, calculate the spectral index that fits them.
    /// Uses only Stokes I.
    pub(super) fn calc_spec_index(&self, fd2: &Self) -> f64 {
        (fd2.i / self.i).ln() / (fd2.freq / self.freq).ln()
    }
}

impl std::ops::Mul<f64> for FluxDensity {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        FluxDensity {
            freq: self.freq,
            i: self.i * rhs,
            q: self.q * rhs,
            u: self.u * rhs,
            v: self.v * rhs,
        }
    }
}

/// This is only used for serialisation
// https://stackoverflow.com/questions/53900612/how-do-i-avoid-generating-json-when-serializing-a-value-that-is-null-or-a-defaul
#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_zero(num: &f64) -> bool {
    num.abs() < f64::EPSILON
}

/// Enum for determining what type of Flux Density the component is
/// Could be list of FluxDensity, power_law or curved_power_law
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FluxDensityType {
    List(Vec1<FluxDensity>),

    PowerLaw {
        /// Spectral index
        si: f64,

        /// Flux Density
        fd: FluxDensity,
    },

    CurvedPowerLaw {
        /// Spectral index
        si: f64,

        /// Flux Density
        fd: FluxDensity,

        /// Spectral curvature
        q: f64,
    },
}

impl FluxDensityType {
    /// Given flux density information, estimate the flux density at a
    /// particular frequency. For power laws / curved power laws, the "ratio" of
    /// the reference frequency and the specified frequencies is used to scale
    /// the reference flux density.
    ///
    /// If enum variant is FluxDensityType::List, then the entries must be
    /// sorted by frequency (which should be the case if the source list was
    /// read by hyperdrive). The estimated flux density is based off of the
    /// Stokes I component, so any other Stokes parameters may be poorly
    /// estimated.
    pub(crate) fn estimate_at_freq(&mut self, freq_hz: f64) -> FluxDensity {
        match self {
            FluxDensityType::PowerLaw { si, fd } => {
                let ratio = calc_flux_ratio(freq_hz, fd.freq, *si);
                let mut new_fd = *fd * ratio;
                new_fd.freq = freq_hz;
                new_fd
            }

            FluxDensityType::CurvedPowerLaw { si, fd, q } => {
                let mut power_law_component = *fd * calc_flux_ratio(freq_hz, fd.freq, *si);
                power_law_component.freq = freq_hz;
                let curved_component = (*q * (freq_hz / fd.freq).ln().powi(2)).exp();
                power_law_component * curved_component
            }

            FluxDensityType::List(fds) => {
                // `smaller_flux_density` is a bad name given to the component's flux
                // density corresponding to a frequency smaller than but nearest to the
                // specified frequency.
                let (spec_index, smaller_flux_density) = {
                    // If there's only one source component, then we must assume the
                    // spectral index for extrapolation.
                    if fds.len() == 1 {
                        (DEFAULT_SPEC_INDEX, &fds[0])
                    }
                    // Otherwise, find the two closest `FluxDensity`s closest to
                    // the given frequency. We enforce that the input list of
                    // `FluxDensity`s is sorted by frequency.
                    else {
                        let mut pair: (&FluxDensity, &FluxDensity) = (&fds[0], &fds[1]);
                        for window in fds.windows(2) {
                            // Bail if the frequencies are out of order.
                            if window[1].freq < window[0].freq {
                                panic!("The list of flux densities used for estimation were not sorted");
                            }

                            pair = (&window[0], &window[1]);

                            // If either element's and the specified freq are the same...
                            if (window[0].freq - freq_hz).abs() < 1e-3 {
                                // ... then just return the flux density
                                // information from this frequency.
                                return window[0];
                            }
                            if (window[1].freq - freq_hz).abs() < 1e-3 {
                                return window[1];
                            }
                            // If the specified freq is smaller than the second
                            // element...
                            if freq_hz < window[1].freq {
                                // ... we're done.
                                break;
                            }
                        }

                        // We now have the two relevant flux densities (on
                        // either side of the target frequency, or the two
                        // closest to the target frequency). If one is positive
                        // and one negative, we have to use a linear fit, not a
                        // spectral index.
                        let (fd1, fd2) = pair;
                        if pair.0.i.signum() != pair.1.i.signum() {
                            let i_rise = fd2.i - fd1.i;
                            let q_rise = fd2.q - fd1.q;
                            let u_rise = fd2.u - fd1.u;
                            let v_rise = fd2.v - fd1.v;
                            let run = fd2.freq - fd1.freq;
                            let i_slope = i_rise / run;
                            let q_slope = q_rise / run;
                            let u_slope = u_rise / run;
                            let v_slope = v_rise / run;
                            return FluxDensity {
                                freq: freq_hz,
                                i: fd1.i + i_slope * (freq_hz - fd1.freq),
                                q: fd1.q + q_slope * (freq_hz - fd1.freq),
                                u: fd1.u + u_slope * (freq_hz - fd1.freq),
                                v: fd1.v + v_slope * (freq_hz - fd1.freq),
                            };
                        }

                        let spec_index = fd1.calc_spec_index(fd2);

                        (
                            spec_index,
                            // If our last component's frequency is smaller than the specified
                            // freq., then we should use that for flux densities.
                            if fd2.freq < freq_hz { fd2 } else { fd1 },
                        )
                    }
                };

                // Now scale the flux densities given the calculated
                // spectral index.
                let flux_ratio = calc_flux_ratio(freq_hz, smaller_flux_density.freq, spec_index);
                FluxDensity {
                    freq: freq_hz,
                    ..*smaller_flux_density
                } * flux_ratio
            }
        }
    }
}

/// Given a spectral index, determine the flux-density ratio of two frequencies.
pub(crate) fn calc_flux_ratio(desired_freq_hz: f64, cat_freq_hz: f64, spec_index: f64) -> f64 {
    (desired_freq_hz / cat_freq_hz).powf(spec_index)
}
