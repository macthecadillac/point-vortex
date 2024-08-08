use chrono::Local;
use clap::Parser;
use npyz::WriterBuilder;
use rayon::prelude::*;

use nalgebra::matrix;

use std::cmp::PartialOrd;
use std::fs::File;
use std::io;
use std::path::Path;

use time_stepper::error;
use time_stepper::problem;
use time_stepper::problem::{Problem, Vector};

use main_error::MainError;

mod config;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to configuration file
    config: String,
    #[arg(long)]
    /// Do not write output to disk
    nosave: bool,
}

struct FiniteTimeLyapunovExponent {
    t: f64,
    tmax: f64,
    delta_t: f64,
    delta: f64,
    prev: f64,
    curr: f64,
    time_stepper: problem::Solver
}

impl FiniteTimeLyapunovExponent {
    fn new(problem: &crate::config::P) -> Self {
        let t = 0.;
        let tmax = problem.t;
        let delta_t = problem.time_step;
        let delta = problem.delta;
        let prev = 0.;
        let curr = 0.;
        let time_stepper = problem::Solver::new(problem);
        Self { t, tmax, delta_t, delta, prev, curr, time_stepper }
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

    fn step(&mut self) -> Result<(), &'static str> {
        self.time_stepper.step();
        self.t += self.delta_t;
        self.prev = self.curr;
        self.curr = FiniteTimeLyapunovExponent::grid_point(
            self.delta,
            self.t,
            &self.time_stepper.state().passive_tracers
        );
        Ok(())
    }

    fn compute(&mut self) -> Result<f64, &'static str> {
        loop {
            let res = self.step();
            if let Err(e) = res { break Err(e) }
            if self.t >= self.tmax { break Ok(self.curr) }
        }
    }
}

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;
    let npt = problem.grid_points.len();
    let problem_w_delta = problem.add_delta_tracers();
    let start_time = Local::now();
    println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

    let mut fbuf = (!args.nosave)
        .then(|| File::create(path.with_extension("npy")).map(|f| io::BufWriter::new(f)))
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
