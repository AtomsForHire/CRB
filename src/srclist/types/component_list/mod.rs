//! An alternative to ['SourceList'] (not the hyperdrive implementation).
//! Follows the original python implementation of the CRB code a bit more.

use super::{FluxDensity, FluxDensityType, SourceComponent, SourceList};
use marlu::RADec;
use std::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Clone, Debug)]
pub struct ComponentList(Vec<SourceComponent>);

impl ComponentList {
    /// Create a component list from an exisiting source_list
    pub(crate) fn new(source_list: SourceList) -> ComponentList {
        let mut component_list: Vec<SourceComponent> = vec![];

        for comp in source_list
            .iter()
            .flat_map(|(_, src)| src.components.iter())
        {
            component_list.push(comp.clone());
        }

        return ComponentList(component_list);
    }

    /// Veto sources by the minimum flux
    pub(crate) fn veto_by_flux(&mut self, noise: f64) {
        self.retain(|comp| {
            // if let FluxDensityType::List(_x) = &comp.flux_type {
            //     eprintln!("LIST COMPONENT DETECTED, I DON'T KNOW HOW TO DEAL WITH THESE YET!");
            //     std::process::exit(1);
            // }
            //
            // let fd = comp.estimate_at_freq(freq);
            // return fd.i > noise;
            match comp.flux_type {
                FluxDensityType::PowerLaw {
                    fd: FluxDensity { i, .. },
                    ..
                } => return i > noise,
                FluxDensityType::CurvedPowerLaw {
                    fd: FluxDensity { i, .. },
                    ..
                } => return i > noise,
                FluxDensityType::List { .. } => return false,
            }
        });
    }

    /// Veto sources by fov
    pub(crate) fn veto_by_fov(&mut self, phase_centre: RADec, lambda: f64, D: f64) {
        return self.retain(|comp| {
            let fov = lambda / D;
            let lmn = comp.radec.to_lmn(phase_centre);

            if (lmn.l.powi(2) + lmn.m.powi(2)).sqrt() < (fov / 2.0f64).sin() {
                return true;
            } else {
                return false;
            }
        });
    }

    pub(crate) fn slice_to_struct(&self, range: std::ops::Range<usize>) -> Self {
        Self(self.0[range].to_vec())
    }

    /// Function for calculating fd's at current frequency so
    /// that it is not done in the calculate_crb function
    pub fn fd_for_freq(&mut self, freq: f64) {
        self.iter_mut().for_each(|comp| {
            let mut temp = comp.clone();

            match comp.flux_type {
            FluxDensityType::PowerLaw {
                si: _,
                ref mut fd,
            } => {
                    *fd = temp.flux_type.estimate_at_freq(freq);
            },
            FluxDensityType::CurvedPowerLaw {
                si: _,
                ref mut fd,
                q: _,
            } => *fd = temp.flux_type.estimate_at_freq(freq),
            FluxDensityType::List(_) => {
                eprintln!("TRYING TO ESTIMATE FOR COMPONENT WITH LIST OF FREQUENCIES WHICH I DUNNO HOW TO HANDLE");
                std::process::exit(1);
            }
            }});
    }

    pub fn get_intensity_list(&self) -> Vec<f64> {
        let result: Vec<f64> = self
            .iter()
            .map(|comp| match comp.flux_type {
                FluxDensityType::PowerLaw {
                    si: _,
                    fd: FluxDensity { freq, i, q, u, v },
                } => i,
                FluxDensityType::CurvedPowerLaw {
                    si,
                    fd: FluxDensity { freq, i, q, u, v },
                    q: _,
                } => i,
                FluxDensityType::List(_) => {
                    eprintln!("TRYING TO GET INTENSITY OF LIST COMPONENT");
                    std::process::exit(1);
                }
            })
            .collect();

        return result;
    }

    pub fn get_lmn_list(&self, phase_centre: RADec) -> Vec<Vec<f64>> {
        let result: Vec<Vec<f64>> = self
            .iter()
            .map(|src| {
                vec![
                    src.radec.to_lmn(phase_centre).l,
                    src.radec.to_lmn(phase_centre).m,
                ]
            })
            .collect();

        return result;
    }
}

// Need these to expose the iter() functionality of Vec
impl Deref for ComponentList {
    type Target = Vec<SourceComponent>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ComponentList {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<std::ops::Range<usize>> for ComponentList {
    type Output = [SourceComponent];

    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<std::ops::Range<usize>> for ComponentList {
    fn index_mut(&mut self, index: std::ops::Range<usize>) -> &mut Self::Output {
        &mut self.0[index]
    }
}
