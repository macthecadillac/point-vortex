use chrono::Local;
use clap::Parser;
use nalgebra::matrix;
use npyz::WriterBuilder;
use rayon::prelude::*;
use serde::Deserialize;

use std::cmp::PartialOrd;
use std::fs::File;
use std::io;
use std::path::PathBuf;

use crate::kernel;
use crate::kernel::{PointVortex, Problem, Vector};

use main_error::MainError;

#[derive(Deserialize)]
#[derive(Clone)]
pub struct P {
    pub sqg: bool,
    pub rossby: f64,
    pub t: f64,
    pub time_step: f64,
    pub delta: f64,
    pub point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "crate::config::deserialize_grid")]
    pub grid_points: Vec<Vector>
}

impl P {
    pub(crate) fn add_delta_tracers(&self) -> Self {
        let mut grid_points = vec![];
        for &v in self.grid_points.iter() {
            grid_points.push(Vector { x: v.x + self.delta, ..v });
            grid_points.push(Vector { x: v.x - self.delta, ..v });
            grid_points.push(Vector { y: v.y + self.delta, ..v });
            grid_points.push(Vector { y: v.y - self.delta, ..v });
            grid_points.push(Vector { z: v.z + self.delta, ..v });
            grid_points.push(Vector { z: v.z - self.delta, ..v });
        }
        Self { grid_points, ..self.clone() }
    }
}

impl Problem for P {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self {
        Self {
            sqg: self.sqg,
            t: self.t,
            rossby: self.rossby,
            time_step: self.time_step,
            delta: self.delta,
            grid_points: tracers.to_owned(),
            point_vortices: self.point_vortices.clone(),
        }
    }
    fn passive_tracers(&self) -> &[Vector] { &self.grid_points }
}

struct FiniteTimeLyapunovExponent {
    t: f64,
    tmax: f64,
    delta_t: f64,
    delta: f64,
    time_stepper: kernel::Solver
}

impl FiniteTimeLyapunovExponent {
    fn new(problem: &P) -> Self {
        let t = 0.;
        let tmax = problem.t;
        let delta_t = problem.time_step;
        let delta = problem.delta;
        let time_stepper = kernel::Solver::new(problem);
        Self { t, tmax, delta_t, delta, time_stepper }
    }

    fn grid_point(delta: f64, t: f64, xs: &[Vector]) -> f64 {
        // Compute Jacobian
        let a = 0.5 / delta;
        let j00 = a * (xs[1].x - xs[0].x);
        let j01 = a * (xs[1].y - xs[0].y);
        let j02 = a * (xs[1].z - xs[0].z);
        let j10 = a * (xs[3].x - xs[2].x);
        let j11 = a * (xs[3].y - xs[2].y);
        let j12 = a * (xs[3].z - xs[2].z);
        let j20 = a * (xs[5].x - xs[4].x);
        let j21 = a * (xs[5].y - xs[4].y);
        let j22 = a * (xs[5].z - xs[4].z);
        let j = matrix![j00, j01, j02; j10, j11, j12; j20, j21, j22];
        let cauchy_green = j * j.transpose();  // nalgebra does dot product this way
        let eigs = cauchy_green.complex_eigenvalues();
        let max_e = eigs.into_iter().max_by(|&a, &b| a.re.partial_cmp(&b.re).unwrap());
        let res = max_e.unwrap().re.ln() / t;
        res
    }

    fn step(&mut self) {
        self.time_stepper.step();
        self.t += self.delta_t;
    }

    fn compute(&mut self) -> Result<f64, &'static str> {
        loop {
            self.step();
            if self.t >= self.tmax {
                let ftle = FiniteTimeLyapunovExponent::grid_point(
                    self.delta,
                    self.t,
                    &self.time_stepper.state().passive_tracers
                );
                break Ok(ftle)
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
}

impl Parameters {
    pub(crate) fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let problem = P::parse(&config_path)?;
        let npt = problem.grid_points.len();
        let problem_w_delta = problem.add_delta_tracers();
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

        let mut fbuf = (!self.nosave)
            .then(|| File::create(config_path.with_extension("npy")).map(|f| io::BufWriter::new(f)))
            .transpose()?;

        if let Some(mut writer) = fbuf.as_mut()
            .map(|b| npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[npt as u64])
                .writer(b).begin_nd())
            .transpose()? {

            let mut solvers: Vec<_> = problem_w_delta.divide(npt)
                .iter()
                .map(FiniteTimeLyapunovExponent::new)
                .collect();
            let ftle_res: Result<Vec<f64>, &'static str> = solvers.par_iter_mut()
                .map(|solver| solver.compute())
                .collect();
            let ftle = ftle_res?;
            writer.extend(ftle.into_iter())?;
            writer.finish()?;
            println!("Done.");
        }
        Ok(())
    }
}
