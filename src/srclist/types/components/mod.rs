// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Sky-model component types
//! - ra: ...
//!   dec: ...
//!   comp_type: ...
//!     ...
//!   flux_type: ...
//!     ...

use super::{FluxDensity, FluxDensityType};
use marlu::RADec;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceComponent {
    #[serde(flatten)]
    pub radec: RADec,

    pub comp_type: ComponentType,

    pub flux_type: FluxDensityType,
}

impl SourceComponent {
    /// Estimate the flux density of this component at a frequency.
    pub(crate) fn estimate_at_freq(&mut self, freq_hz: f64) -> FluxDensity {
        self.flux_type.estimate_at_freq(freq_hz)
    }

    /// Is this component a point source?
    pub(crate) fn is_point(&self) -> bool {
        self.comp_type.is_point()
    }

    /// Is this component a gaussian source?
    pub(crate) fn is_gaussian(&self) -> bool {
        self.comp_type.is_gaussian()
    }

    /// Is this component a shapelet source?
    pub(crate) fn is_shapelet(&self) -> bool {
        self.comp_type.is_shapelet()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComponentType {
    Point,

    Gaussian {
        maj: f64,
        min: f64,
        pa: f64,
    },

    Shapelet {
        maj: f64,

        min: f64,

        pa: f64,

        coeffs: Box<[ShapeletCoeff]>,
    },
}

impl ComponentType {
    pub(crate) fn is_point(&self) -> bool {
        return matches!(self, Self::Point);
    }

    pub(crate) fn is_gaussian(&self) -> bool {
        return matches!(self, Self::Gaussian { .. });
    }

    pub(crate) fn is_shapelet(&self) -> bool {
        return matches!(self, Self::Shapelet { .. });
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeletCoeff {
    pub n1: u8,
    pub n2: u8,
    pub value: f64,
}
//////////////////////////////////////////////////////////////////////////
// #[derive(Clone, Debug, Default)]
// pub(crate) struct PointComponentParams {
//     pub(crate) radecs: Vec<RADec>,
//     pub(crate) lmns: Vec<LmnRime>,
//     /// Instrumental (i.e. XX, XY, YX, XX).
//     pub(crate) flux_densities: Array2<Jones<f64>>,
// }
//
// /// Gaussian-source-component parameters.
// ///
// /// See the doc comment for [PointComponentParams] for more info.
// #[derive(Clone, Debug, Default)]
// pub(crate) struct GaussianComponentParams {
//     pub(crate) radecs: Vec<RADec>,
//     pub(crate) lmns: Vec<LmnRime>,
//     /// Instrumental (i.e. XX, XY, YX, XX).
//     pub(crate) flux_densities: Array2<Jones<f64>>,
//     pub(crate) gaussian_params: Vec<GaussianParams>,
// }
//
// /// Shapelet-source-component parameters.
// ///
// /// See the doc comment for [PointComponentParams] for more info.
// #[derive(Clone, Debug, Default)]
// pub(crate) struct ShapeletComponentParams {
//     pub(crate) radecs: Vec<RADec>,
//     pub(crate) lmns: Vec<LmnRime>,
//     /// Instrumental (i.e. XX, XY, YX, XX).
//     pub(crate) flux_densities: Array2<Jones<f64>>,
//     pub(crate) gaussian_params: Vec<GaussianParams>,
//     pub(crate) shapelet_coeffs: Vec<Vec<ShapeletCoeff>>,
// }
//
// /// Major and minor axes as well as a positional angle to describe a Gaussian
// /// (or something like a Gaussian, e.g. a shapelet).
// #[derive(Clone, Debug, PartialEq)]
// pub(crate) struct GaussianParams {
//     /// Major axis size \[radians\]
//     pub(crate) maj: f64,
//     /// Minor axis size \[radians\]
//     pub(crate) min: f64,
//     /// Position angle \[radians\]
//     pub(crate) pa: f64,
// }
//
// pub(crate) struct ComponentList {
//     pub(crate) points: PointComponentParams,
//     pub(crate) gaussians: GaussianComponentParams,
//     pub(crate) shapelets: ShapeletComponentParams,
// }
//
// impl ComponentList {
//     pub(crate) fn new(
//         source_list: &SourceList,
//         phase_centre: RADec,
//     ) -> ComponentList {
//         // Unpack each of the component parameters into vectors.
//         let mut point_radecs = vec![];
//         let mut point_lmns = vec![];
//         let mut point_fds: Vec<FluxDensityType> = vec![];
//
//         let mut gaussian_radecs = vec![];
//         let mut gaussian_lmns = vec![];
//         let mut gaussian_fds: Vec<FluxDensityType> = vec![];
//         let mut gaussian_gaussian_params = vec![];
//
//         let mut shapelet_radecs = vec![];
//         let mut shapelet_lmns = vec![];
//         let mut shapelet_fds: Vec<FluxDensityType> = vec![];
//         let mut shapelet_gaussian_params = vec![];
//         let mut shapelet_coeffs: Vec<Vec<ShapeletCoeff>> = vec![];
//
//         // Reverse the source list; if the source list has been sorted
//         // (brightest sources first), reversing makes the dimmest sources get
//         // used first. This is good because floating-point precision errors are
//         // smaller when similar values are accumulated. Accumulating into a
//         // float starting from the brightest component means that the
//         // floating-point precision errors are greater as we work through the
//         // source list.
//         for comp in source_list
//             .iter()
//             .rev()
//             .flat_map(|(_, src)| src.components.iter())
//         {
//             let comp_lmn = comp.radec.to_lmn(phase_centre).prepare_for_rime();
//             match &comp.comp_type {
//                 ComponentType::Point => {
//                     point_radecs.push(comp.radec);
//                     point_lmns.push(comp_lmn);
//                     point_fds.push(comp.flux_type.clone());
//                 }
//
//                 ComponentType::Gaussian { maj, min, pa } => {
//                     gaussian_radecs.push(comp.radec);
//                     gaussian_lmns.push(comp_lmn);
//                     gaussian_fds.push(comp.flux_type.clone());
//                     gaussian_gaussian_params.push(GaussianParams {
//                         maj: *maj,
//                         min: *min,
//                         pa: *pa,
//                     });
//                 }
//
//                 ComponentType::Shapelet {
//                     maj,
//                     min,
//                     pa,
//                     coeffs,
//                 } => {
//                     shapelet_radecs.push(comp.radec);
//                     shapelet_lmns.push(comp_lmn);
//                     shapelet_fds.push(comp.flux_type.clone());
//                     shapelet_gaussian_params.push(GaussianParams {
//                         maj: *maj,
//                         min: *min,
//                         pa: *pa,
//                     });
//                     shapelet_coeffs.push(coeffs.to_vec());
//                 }
//             }
//         }
//
//             return Self {
//             points: PointComponentParams {
//                 radecs: point_radecs,
//                 lmns: point_lmns,
//                 flux_densities: point_fds,
//             },
//             gaussians: GaussianComponentParams {
//                 radecs: gaussian_radecs,
//                 lmns: gaussian_lmns,
//                 flux_densities: gaussian_fds,
//                 gaussian_params: gaussian_gaussian_params,
//             },
//             shapelets: ShapeletComponentParams {
//                 radecs: shapelet_radecs,
//                 lmns: shapelet_lmns,
//                 flux_densities: shapelet_fds,
//                 gaussian_params: shapelet_gaussian_params,
//                 shapelet_coeffs,
//             },
//         }
//     }
// }
