use chrono::Local;
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;

use std::path::PathBuf;

use crate::config::Parse;
use crate::kernel;
use crate::kernel::{PointVortex, Specification, Vector};
use crate::utils;

use main_error::MainError;

#[derive(Deserialize)]
#[derive(Clone)]
pub struct SimulationSpecification {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    #[serde(deserialize_with = "crate::config::grid_or_distances")]
    pub point_vortices: Vec<[PointVortex; 3]>,
}

impl Parse for SimulationSpecification {}

impl SimulationSpecification {
    pub fn divide(&self) -> Vec<SubSpec> {
        let pv = &self.point_vortices;
        let &SimulationSpecification { sqg, rossby, duration, time_step, .. } = self;
        pv.iter()
          .map(|&point_vortices| SubSpec { sqg, rossby, duration, time_step, point_vortices })
          .collect()
    }
}

#[derive(Clone)]
pub struct SubSpec {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub point_vortices: [PointVortex; 3],
}

impl Specification for SubSpec {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &[] }
    fn replace_tracers(&self, _: &[Vector]) -> Self { self.clone() }
}

#[derive(Debug, Default, Clone, Copy)]
#[derive(Deserialize)]
#[derive(npyz::AutoSerialize, npyz::Serialize)]
pub struct Distances { pub r12: f64, pub r23: f64, pub r31: f64 }

#[derive(Debug, Default, Clone, Copy)]
#[derive(Deserialize)]
#[derive(npyz::AutoSerialize, npyz::Serialize)]
pub struct ScaledDistances { pub r12: f64, pub r23: f64 }

impl Distances {
    fn section(self, other: Distances) -> Option<ScaledDistances> {
        if self.r31 > 1. && other.r31 < 1. {
            let r12 = (other.r31 * self.r12 + self.r31 * other.r12) / (self.r31 + other.r31);
            let r23 = (other.r31 * self.r23 + self.r31 * other.r23) / (self.r31 + other.r31);
            Some(ScaledDistances { r12, r23 })
        } else {
            None
        }
    }
}

struct DistancePoincareSection {
    t: f64,
    duration: f64,
    delta_t: f64,
    intersections: Vec<ScaledDistances>,
    prev_state: Distances,
    time_stepper: kernel::TimeStepper
}

impl DistancePoincareSection {
    fn new(spec: &SubSpec) -> Self {
        let t = 0.;
        let duration = spec.duration;
        let delta_t = spec.time_step;
        let intersections = vec![];
        let time_stepper = kernel::TimeStepper::new(spec);
        let last_pv_pos = &time_stepper.state().point_vortices;
        let r12 = (last_pv_pos[0].position - last_pv_pos[1].position).norm_pow(0.5);
        let r23 = (last_pv_pos[1].position - last_pv_pos[2].position).norm_pow(0.5);
        let r31 = (last_pv_pos[2].position - last_pv_pos[0].position).norm_pow(0.5);
        let prev_state = Distances { r12, r23, r31 };
        Self { t, duration, delta_t, intersections, prev_state, time_stepper }
    }

    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
        let pv_pos = &self.time_stepper.state().point_vortices;
        let r12 = (pv_pos[0].position - pv_pos[1].position).norm_pow(0.5);
        let r23 = (pv_pos[1].position - pv_pos[2].position).norm_pow(0.5);
        let r31 = (pv_pos[2].position - pv_pos[0].position).norm_pow(0.5);
        let state = Distances { r12, r23, r31 };
        self.intersections.extend(self.prev_state.section(state));
        self.prev_state = state;
    }

    fn compute(mut self) -> Vec<ScaledDistances> {
        loop {
            self.step();
            if self.t >= self.duration {
                break self.intersections
            }
        }
    }
}

#[derive(Parser, Debug)]
pub struct Parameters {
    /// Path to configuration file
    pub config: PathBuf,
    #[arg(long)]
    /// Do not write output to disk
    pub nosave: bool,
}

impl Parameters {
    pub fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let spec = SimulationSpecification::parse(&config_path)?;
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

        let poincare_sections: Vec<_> = spec.divide()
            .into_par_iter()
            .flat_map(|solver| DistancePoincareSection::new(&solver).compute())
            .collect();

        let npz_path = config_path.with_extension("npz");
        let mut npz = utils::Writer::new(&npz_path)?;
        npz.writez("sqg", &[1], [spec.sqg])?;
        npz.writez("rossby", &[1], [spec.rossby])?;
        npz.writez("duration", &[1], [spec.duration])?;
        npz.writez("time_step", &[1], [spec.time_step])?;
        npz.writez("sections", &[poincare_sections.len() as u64], &poincare_sections)?;

        Ok(())
    }
}
