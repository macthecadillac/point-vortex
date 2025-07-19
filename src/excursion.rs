use chrono::Local;
use clap::Parser;
use main_error::MainError;
use npyz::WriterBuilder;
use rayon::prelude::*;
use serde::Deserialize;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;

use crate::config::Parse;
use crate::error;
use crate::kernel;
use crate::kernel::{PointVortex, Specification, Vector};
use crate::utils;

#[derive(Deserialize)]
#[derive(Clone)]
struct SimulationSpecification {
    sqg: bool,
    rossby: f64,
    duration: f64,
    time_step: f64,
    point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "crate::config::grid_or_vectors")]
    passive_tracers: Vec<Vector>,
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

#[derive(serde::Deserialize)]
#[derive(npyz::AutoSerialize, npyz::Serialize)]
struct VerticalExcursion {
    loc: Vector,
    excursion: f64
}

#[derive(Copy, Clone)]
struct VerticalMinMax {
    loc: Vector,
    max: f64,
    min: f64
}

impl From<VerticalMinMax> for VerticalExcursion {
    fn from(source: VerticalMinMax) -> VerticalExcursion {
        VerticalExcursion {
            loc: source.loc,
            excursion: source.max - source.min
        }
    }
}

#[derive(Debug, Parser)]
pub struct Parameters {
    /// Path to configuration file
    pub config: PathBuf,
    #[arg(long)]
    /// Number of threads. Runs in single-threaded mode if not provided
    pub nthreads: Option<usize>
}

impl Parameters {
    fn base_op(time_stepper: &kernel::TimeStepper, mbuf: &mut [VerticalMinMax]) {
        for (pt, ve) in time_stepper.state().passive_tracers.iter()
                                   .zip(mbuf.iter_mut()) {
            *ve = VerticalMinMax {
                max: pt.z.max(ve.max),
                min: pt.z.min(ve.min),
                ..*ve
            };
        }
    }

    pub fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let spec = SimulationSpecification::parse(&config_path)?;
        let niter = (spec.duration as f64 / spec.time_step).round() as usize;
        let npt = spec.passive_tracers.len();
        let nthreads = self.nthreads.unwrap_or(1);
        if npt % nthreads > 0 { Err(error::Error::NThreadsError)? }
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

        let fbuf = File::create(config_path.with_extension("npy"))
                                           .map(|f| BufWriter::new(f))?;
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[npt as u64])
            .writer(fbuf).begin_nd()?;

        if nthreads == 1 {
            let mut time_stepper = kernel::TimeStepper::new(&spec);
            let mut mbuf: Vec<_> = time_stepper.state().passive_tracers.iter()
                .map(|&v| VerticalMinMax { loc: v, min: v.z, max: v.z })
                .collect();
            let mut progress = utils::Progress::new(niter);
            for _ in 1..niter {
                time_stepper.step();
                Parameters::base_op(&time_stepper, &mut mbuf);
                progress.step(true);
            }
            writer.extend(mbuf.into_iter().map(VerticalExcursion::from))?;
        } else {
            let buf_size_per_thread = npt / nthreads;
            let mut mbufs: Vec<Vec<VerticalMinMax>> = vec![vec![]; nthreads];
            for mbuf in mbufs.iter_mut() {
                mbuf.reserve_exact(buf_size_per_thread);
            }
            let mut time_steppers = Vec::new();
            for (p, mbuf) in spec.divide(nthreads).into_iter().zip(mbufs.iter_mut()) {
                let time_stepper = kernel::TimeStepper::new(&p);
                mbuf.extend(time_stepper.state().passive_tracers.iter()
                    .map(|&v| VerticalMinMax { loc: v, min: v.z, max: v.z }));
                time_steppers.push(time_stepper);
            }
            let mut progress = vec![utils::Progress::new(niter); nthreads];
            mbufs.par_iter_mut()
                .zip(time_steppers.par_iter_mut())
                .zip(progress.par_iter_mut())
                .enumerate()
                .for_each(|(n, ((mbuf, time_stepper), progress))| {
                    for _ in 1 ..niter {
                        time_stepper.step();
                        Parameters::base_op(&time_stepper, mbuf);
                        progress.step(n == 0);
                    }
                });
            for mbuf in mbufs.into_iter() {
                writer.extend(mbuf.into_iter().map(VerticalExcursion::from))?;
            }
        }
        writer.finish()?;
        Ok(())
    }
}
