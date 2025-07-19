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

use crate::triangle_ps::Distances;

#[derive(Deserialize)]
#[derive(Clone)]
struct SimulationSpecification {
    sqg: bool,
    rossby: f64,
    duration: f64,
    time_step: f64,
    #[serde(deserialize_with = "crate::config::grid_or_distances")]
    point_vortices: Vec<[PointVortex; 3]>,
    write_interval: Option<usize>
}

impl Parse for SimulationSpecification {}

impl SimulationSpecification {
    fn divide(&self) -> Vec<SubSpec> {
        let pv = &self.point_vortices;
        let &SimulationSpecification { sqg, rossby, duration, time_step, write_interval, .. } = self;
        pv.iter()
          .map(|&point_vortices| SubSpec { sqg, rossby, duration, time_step, point_vortices, write_interval })
          .collect()
    }
}

#[derive(Clone)]
struct SubSpec {
    sqg: bool,
    rossby: f64,
    duration: f64,
    time_step: f64,
    point_vortices: [PointVortex; 3],
    write_interval: Option<usize>
}

impl Specification for SubSpec {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &[] }
    fn replace_tracers(&self, _: &[Vector]) -> Self { self.clone() }
}

struct Counter { write_interval: usize, counter: usize }

struct Simulation {
    t: f64,
    duration: f64,
    delta_t: f64,
    data: Vec<Distances>,
    time_stepper: kernel::TimeStepper,
    counter: Option<Counter>
}

impl Simulation {
    fn new(spec: &SubSpec) -> Self {
        let t = 0.;
        let duration = spec.duration;
        let delta_t = spec.time_step;
        let time_stepper = kernel::TimeStepper::new(spec);
        let last_pv_pos = &time_stepper.state().point_vortices;
        let r12 = (last_pv_pos[0].position - last_pv_pos[1].position).norm_pow(0.5);
        let r23 = (last_pv_pos[1].position - last_pv_pos[2].position).norm_pow(0.5);
        let r31 = (last_pv_pos[2].position - last_pv_pos[0].position).norm_pow(0.5);
        let init_state = Distances { r12, r23, r31 };
        let data = vec![init_state];
        let counter = spec.write_interval.map(|write_interval| Counter { write_interval, counter: 0 });
        Self { t, duration, delta_t, data, counter, time_stepper }
    }

    fn step(&mut self) {
        if let Some(Counter { write_interval, counter }) = self.counter.as_mut() {
            *counter = (*counter + 1) % *write_interval
        }
        self.time_stepper.step();
        self.t += self.delta_t;
        let pv_pos = &self.time_stepper.state().point_vortices;
        let r12 = (pv_pos[0].position - pv_pos[1].position).norm_pow(0.5);
        let r23 = (pv_pos[1].position - pv_pos[2].position).norm_pow(0.5);
        let r31 = (pv_pos[2].position - pv_pos[0].position).norm_pow(0.5);
        let state = Distances { r12, r23, r31 };
        if let Some(Counter { counter: 0, .. }) = self.counter {
            self.data.push(state)
        } else if self.counter.is_none() {
            self.data.push(state)
        }
    }

    fn compute(mut self) -> Vec<Distances> {
        loop {
            self.step();
            if self.t >= self.duration {
                break self.data
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

        let lengths: Vec<_> = spec.divide()
            .into_par_iter()
            .flat_map(|solver| Simulation::new(&solver).compute())
            .collect();

        let npz_path = config_path.with_extension("npz");
        let mut npz = utils::Writer::new(&npz_path)?;
        npz.writez("sqg", &[1], [spec.sqg])?;
        npz.writez("rossby", &[1], [spec.rossby])?;
        npz.writez("duration", &[1], [spec.duration])?;
        npz.writez("time_step", &[1], [spec.time_step])?;
        npz.writez("lengths", &[lengths.len() as u64], &lengths)?;

        Ok(())
    }
}
