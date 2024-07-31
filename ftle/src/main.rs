use chrono::Local;
use clap::Parser;
use npyz::WriterBuilder;
use rayon::prelude::*;

use ndarray::array;
use ndarray_linalg::Eig;

use std::cmp::PartialOrd;
use std::fs::File;
use std::io;
//use std::io::prelude::*;
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

struct ForwardTimeLyapunovExponent<'a> {
    t: f64,
    delta_t: f64,
    delta: f64,
    prev: f64,
    curr: f64,
    tol: f64,
    time_stepper: problem::Solver<'a>,
    done: bool
}

impl<'a> ForwardTimeLyapunovExponent<'a> {
    fn new(problem: &crate::config::P,
           pv_buf: &'a mut [problem::PointVortex],
           v_buf: &'a mut [Vector]) -> Self {
        let t = 0.;
        let delta_t = problem.time_step;
        let delta = problem.delta;
        let prev = 0.;
        let curr = 0.;
        let tol = problem.tol;
        let time_stepper = problem::Solver::new(problem, pv_buf, v_buf);
        let done = false;
        Self { t, delta_t, delta, prev, curr, tol, time_stepper, done }
    }

    fn grid_point(delta: f64, t: f64, xs: &[Vector]) -> Result<f64, &'static str> {
        // Compute Jacobian
        let x0 = xs[0];
        let j00 = (xs[1].x - x0.x) / delta;
        let j01 = (xs[1].y - x0.y) / delta;
        let j02 = (xs[1].z - x0.z) / delta;
        let j10 = (xs[2].x - x0.x) / delta;
        let j11 = (xs[2].y - x0.y) / delta;
        let j12 = (xs[2].z - x0.z) / delta;
        let j20 = (xs[3].x - x0.x) / delta;
        let j21 = (xs[3].y - x0.y) / delta;
        let j22 = (xs[3].z - x0.z) / delta;
        let j = array![[j00, j01, j02], [j10, j11, j12], [j20, j21, j22]];
        let cauchy_green = j.dot(&j.t());
        if let Ok((eigs, _)) = cauchy_green.eig() {
            let max_e = eigs.into_iter().max_by(|&a, &b| a.re.partial_cmp(&b.re).unwrap());
            let res = max_e.unwrap().re.ln() / t;
            Ok(res)
        } else {
            Err("Error at eig")
        }
    }

    fn step(&mut self) -> Result<(), &'static str> {
        self.time_stepper.step();
        self.t += self.delta_t;
        self.prev = self.curr;
        self.curr = ForwardTimeLyapunovExponent::grid_point(
            self.delta,
            self.t,
            &self.time_stepper.state().passive_tracers
        )?;
        self.done = (self.prev - self.curr).abs() < self.tol;
        Ok(())
    }

    fn compute(&mut self) -> Result<f64, &'static str> {
        loop {
            let res = self.step();
            if let Err(e) = res { break Err(e) }
            if self.done { break Ok(self.curr) }
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

        let subproblems = problem_w_delta.divide(npt);
        let pv_buf_size = subproblems.iter().map(|p| problem::Solver::pv_buf_size(p)).sum();
        let v_buf_size = subproblems.iter().map(|p| problem::Solver::v_buf_size(p)).sum();
        let mut pv_buf = vec![problem::PointVortex::default(); pv_buf_size];
        let mut v_buf = vec![Vector::default(); v_buf_size];
        let ftle_res: Result<Vec<f64>, &'static str> =
            subproblems.par_iter().zip(pv_buf.par_chunks_mut(pv_buf_size / npt))
                                  .zip(v_buf.par_chunks_mut(v_buf_size / npt))
                                  .map(|((subproblem, pv_buf_), v_buf_)| {
                let mut solver = ForwardTimeLyapunovExponent::new(subproblem, pv_buf_, v_buf_);
                solver.compute()
            })
            .collect();
        let ftle = ftle_res?;
        writer.extend(ftle.into_iter())?;
        writer.finish()?;
        println!("Done.");
    }
    Ok(())
}
