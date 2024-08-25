use chrono::Local;
use clap::Parser;
use rayon::prelude::*;
use serde::Deserialize;

use std::path::PathBuf;

use crate::kernel;
use crate::kernel::{PointVortex, Specification, Vector};
use crate::utils;

use main_error::MainError;

#[derive(Deserialize)]
#[derive(Copy, Clone, Debug)]
// a x + b y + c z + d = 0
struct Plane { a: f64, b: f64, c: f64, d: f64 }

impl Plane {
    fn dist(self, v: Vector) -> f64 {
        self.a * v.x + self.b * v.y + self.c * v.z + self.d
    }

    fn section(self, prev: Vector, next: Vector) -> Option<Vector> {
        let p = self.dist(prev);
        let n = self.dist(next);
        if p > 0. && n < 0. {
            Some((p * next - n * prev) / (p - n))
        } else {
            None
        }
    }
}

#[derive(Deserialize)]
#[derive(Clone)]
struct SimulationSpecification {
    sqg: bool,
    rossby: f64,
    duration: f64,
    time_step: f64,
    plane: Plane,
    point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "crate::config::grid_or_vectors")]
    passive_tracers: Vec<Vector>
}

impl Specification for SimulationSpecification {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &self.passive_tracers }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self { Self { passive_tracers: tracers.to_owned(), ..self.clone() } }
}

#[derive(npyz::AutoSerialize, npyz::Serialize)]
struct Data {
    pv_index: u64,
    location: Vector
}

struct RealSpacePoincareSection {
    t: f64,
    duration: f64,
    delta_t: f64,
    intersections: Vec<Data>,
    pv_index: usize,
    pv: Vec<PointVortex>,
    last_pos: Vec<Vector>,
    plane: Plane,
    time_stepper: kernel::TimeStepper
}

impl RealSpacePoincareSection {
    fn new(spec: &SimulationSpecification) -> Self {
        let t = 0.;
        let duration = spec.duration;
        let delta_t = spec.time_step;
        let intersections = vec![];
        let time_stepper = kernel::TimeStepper::new(spec);
        let last_pos = time_stepper.state().passive_tracers.clone();
        let plane = spec.plane;
        let pv_index = 0;
        let pv = vec![];
        Self { t, duration, delta_t, intersections, plane, pv, pv_index, last_pos, time_stepper }
    }

    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
        let state = self.time_stepper.state();
        let mut append_pv = true;
        for intersection in self.last_pos.iter()
                .zip(state.passive_tracers.iter())
                .flat_map(|(&prev, &next)| self.plane.section(prev, next)) {
            if append_pv {
                self.pv.extend_from_slice(&self.time_stepper.state().point_vortices);
                append_pv = false;
                self.pv_index += 1;
            }
            let data = Data { pv_index: self.pv_index as u64, location: intersection };
            self.intersections.push(data);
        }
        if !append_pv { self.pv_index += 1 };
        self.last_pos.copy_from_slice(&self.time_stepper.state().passive_tracers);
    }

    fn compute(mut self) -> (Vec<PointVortex>, Vec<Data>) {
        loop {
            self.step();
            if self.t >= self.duration {
                break (self.pv, self.intersections)
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
    #[arg(long)]
    /// Number of threads. Runs in single-threaded mode if not provided
    pub nthreads: Option<usize>
}

impl Parameters {
    pub fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let spec = SimulationSpecification::parse(&config_path)?;
        let nthreads = self.nthreads.unwrap_or(1);
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));
        let npv = spec.point_vortices.len();

        let npz_path = config_path.with_extension("npz");

        let (point_vortices, poincare_sections) = if nthreads == 1 {
            let solver = RealSpacePoincareSection::new(&spec);
            solver.compute()
        } else {
            spec.divide(nthreads)
                .into_par_iter()
                .flat_map(|solver| RealSpacePoincareSection::new(&solver).compute())
                .collect()
        };

        let mut npz = utils::Writer::new(&npz_path)?;
        npz.writez("sqg", &[1], [spec.sqg])?;
        npz.writez("rossby", &[1], [spec.rossby])?;
        npz.writez("duration", &[1], [spec.duration])?;
        npz.writez("time_step", &[1], [spec.time_step])?;
        npz.writez("sections", &[poincare_sections.len() as u64], &poincare_sections)?;
        npz.writez("pv", &[(point_vortices.len() / npv) as u64, npv as u64], &point_vortices)?;

        Ok(())
    }
}
