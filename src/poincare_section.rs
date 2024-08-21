use serde::Deserialize;

use chrono::Local;
use clap::Parser;
use npyz::WriterBuilder;
use rayon::prelude::*;

use std::fs::File;
use std::io;
use std::path::PathBuf;

use crate::kernel;
use crate::kernel::{PointVortex, Problem, Vector};

use main_error::MainError;

#[derive(Deserialize)]
#[derive(Copy, Clone, Debug)]
// a x + b y + c z + d = 0
pub struct Plane { a: f64, b: f64, c: f64, d: f64 }

impl Plane {
    fn dist(self, v: Vector) -> f64 {
        self.a * v.x + self.b * v.y + self.c * v.z + self.d
    }

    pub fn section(self, prev: Vector, next: Vector) -> Option<Vector> {
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
pub struct P {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub plane: Plane,
    pub point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "crate::config::grid_or_vectors")]
    pub passive_tracers: Vec<Vector>
}

impl Problem for P {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &self.passive_tracers }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self { Self { passive_tracers: tracers.to_owned(), ..self.clone() } }
}

struct PoincareSection {
    t: f64,
    duration: f64,
    delta_t: f64,
    intersections: Vec<Vector>,
    last_pos: Vec<Vector>,
    plane: Plane,
    time_stepper: kernel::Solver
}

impl PoincareSection {
    fn new(problem: &P) -> Self {
        let t = 0.;
        let duration = problem.duration;
        let delta_t = problem.time_step;
        let intersections = vec![];
        let time_stepper = kernel::Solver::new(problem);
        let last_pos = time_stepper.state().passive_tracers.clone();
        let plane = problem.plane;
        Self { t, duration, delta_t, intersections, plane, last_pos, time_stepper }
    }

    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
        let state = self.time_stepper.state();
        self.intersections.extend(
            self.last_pos.iter()
                .zip(state.passive_tracers.iter())
                .flat_map(|(&prev, &next)| self.plane.section(prev, next))
            );
        self.last_pos.copy_from_slice(&self.time_stepper.state().passive_tracers);
    }

    fn compute(mut self) -> Vec<Vector> {
        loop {
            self.step();
            if self.t >= self.duration {
                break self.intersections
            }
        }
    }
}

#[derive(Parser, Debug)]
pub(crate) struct Parameters {
    /// Path to configuration file
    pub(crate) config: PathBuf,
    #[arg(long)]
    /// Do not write output to disk
    pub(crate) nosave: bool,
    #[arg(long)]
    /// Number of threads. Runs in single-threaded mode if not provided
    pub(crate) nthreads: Option<usize>
}

impl Parameters {
    pub(crate) fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let problem = P::parse(&config_path)?;
        let nthreads = self.nthreads.unwrap_or(1);
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

        let mut fbuf = (!self.nosave)
            .then(|| File::create(config_path.with_extension("npy")).map(|f| io::BufWriter::new(f)))
            .transpose()?;

        let solvers: Vec<_> = problem.divide(nthreads)
            .iter()
            .map(PoincareSection::new)
            .collect();
        let poincare_sections: Vec<_> = solvers.into_par_iter()
            .flat_map(|solver| solver.compute())
            .collect();

        if let Some(mut writer) = fbuf.as_mut()
            .map(|b| npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[poincare_sections.len() as u64])
                .writer(b).begin_nd())
            .transpose()? {

            writer.extend(poincare_sections.into_iter())?;
            writer.finish()?;
            println!("Done.");
        }
        Ok(())
    }
}
