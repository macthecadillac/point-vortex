use chrono::Local;
use clap::Parser;
use num::complex::Complex;
use rayon::prelude::*;
use serde::Deserialize;

use std::path::PathBuf;

use crate::config::Parse;
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
enum AuxilliaryDataType { #[serde(alias = "phase")] Phase2D }

#[derive(Deserialize)]
#[derive(Clone)]
struct SimulationSpecification {
    sqg: bool,
    rossby: f64,
    duration: f64,
    time_step: f64,
    plane: Plane,
    aux_data: Option<AuxilliaryDataType>,
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

impl Parse for SimulationSpecification {}

#[derive(npyz::AutoSerialize, npyz::Serialize)]
struct PhaseVec { phase: f64, location: Vector }

struct RealSpacePoincareSection<T> where T: npyz::AutoSerialize + npyz::Serialize {
    t: f64,
    duration: f64,
    delta_t: f64,
    intersections: Vec<T>,
    last_pos: Vec<Vector>,
    orig_pv_pos: Vec<PointVortex>,
    plane: Plane,
    time_stepper: kernel::TimeStepper
}

impl<T> RealSpacePoincareSection<T> where T: npyz::AutoSerialize + npyz::Serialize {
    fn new(spec: &SimulationSpecification) -> Self {
        let t = 0.;
        let duration = spec.duration;
        let delta_t = spec.time_step;
        let intersections = vec![];
        let time_stepper = kernel::TimeStepper::new(spec);
        let last_pos = time_stepper.state().passive_tracers.clone();
        let orig_pv_pos = time_stepper.state().point_vortices.clone();
        let plane = spec.plane;
        Self { t, duration, delta_t, intersections, plane, last_pos, orig_pv_pos, time_stepper }
    }
}

macro_rules! impl_T {
    ($T: ty) => {
        impl RealSpacePoincareSection<$T> {
            fn compute(mut self) -> Vec<$T> {
                loop {
                    self.step();
                    if self.t >= self.duration {
                        break self.intersections
                    }
                }
            }
        }

        impl RealSpacePoincareSection<$T> {
            fn run(nthreads: usize, spec: &SimulationSpecification) -> Vec<$T> {
                if nthreads == 1 {
                    let solver = RealSpacePoincareSection::<$T>::new(spec);
                    solver.compute()
                } else {
                    spec.divide(nthreads)
                        .into_par_iter()
                        .flat_map(|spec| {
                            let solver = RealSpacePoincareSection::<$T>::new(&spec);
                            solver.compute()
                        })
                        .collect()
                }
            }
        }
    }
}

impl_T!(PhaseVec);
impl_T!(Vector);

impl RealSpacePoincareSection<PhaseVec> {
    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
        let state = self.time_stepper.state();
        for intersection in self.last_pos.iter()
                .zip(state.passive_tracers.iter())
                .flat_map(|(&prev, &next)| self.plane.section(prev, next)) {
            let PointVortex { position: Vector { x, y, .. }, .. } = self.time_stepper.state().point_vortices[0];
            let PointVortex { position: Vector { x: x0, y: y0, .. }, .. } = self.orig_pv_pos[0];
            let phase = (Complex { re: x, im: y } / Complex { re: x0, im: y0 }).arg();
            let data = PhaseVec { phase, location: intersection };
            self.intersections.push(data);
        }
        self.last_pos.copy_from_slice(&self.time_stepper.state().passive_tracers);
    }
}

impl RealSpacePoincareSection<Vector> {
    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
        let state = self.time_stepper.state();
        for intersection in self.last_pos.iter()
                .zip(state.passive_tracers.iter())
                .flat_map(|(&prev, &next)| self.plane.section(prev, next)) {
            self.intersections.push(intersection);
        }
        self.last_pos.copy_from_slice(&self.time_stepper.state().passive_tracers);
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

        let npz_path = config_path.with_extension("npz");

        let mut npz = utils::Writer::new(&npz_path)?;
        npz.writez("sqg", &[1], [spec.sqg])?;
        npz.writez("rossby", &[1], [spec.rossby])?;
        npz.writez("duration", &[1], [spec.duration])?;
        npz.writez("time_step", &[1], [spec.time_step])?;

        match spec.aux_data {
            Some(AuxilliaryDataType::Phase2D) => {
                let poincare_sections = RealSpacePoincareSection::<PhaseVec>::run(nthreads, &spec);
                npz.writez("sections", &[poincare_sections.len() as u64], &poincare_sections)?;
            },
            None => {
                let poincare_sections = RealSpacePoincareSection::<Vector>::run(nthreads, &spec);
                npz.writez("sections", &[poincare_sections.len() as u64], &poincare_sections)?;
            }
        }

        Ok(())
    }
}
