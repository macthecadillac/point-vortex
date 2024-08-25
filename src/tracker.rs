use chrono::Local;
use clap::Parser;
use main_error::MainError;
use npyz::WriterBuilder;
use rayon::prelude::*;
use serde::Deserialize;

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::slice;

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
    write_interval: Option<usize>,
}

impl Specification for SimulationSpecification {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &self.passive_tracers }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self { Self { passive_tracers: tracers.to_owned(), ..self.clone() } }
}

struct MultiBufferData<'a, T> {
    npv: usize,
    data: Vec<(usize, slice::Chunks<'a, T>)>
}

impl<'a, T> MultiBufferData<'a, T> {
    fn from(v: &'a mut [(usize, &'a mut [T])], npv: usize, step_size: usize) -> Self {
        let data = v.iter_mut().map(|(n, buf)| (*n, buf.chunks(step_size))).collect();
        MultiBufferData { npv, data }
    }
}

struct MultiBufferChunksIter<'a, T> {
    index: usize,
    npv: usize,
    data: &'a mut [(usize, slice::Chunks<'a, T>)]
}

impl<'a, T: Clone> MultiBufferData<'a, T> {
    fn chunks(&'a mut self) -> MultiBufferChunksIter<'a, T> {
        MultiBufferChunksIter {
            index: self.data.len() - 1,
            npv: self.npv,
            data: self.data.as_mut_slice()
        }
    }

    fn iter(&'a mut self) -> impl Iterator<Item=T> + 'a {
        self.chunks().flat_map(|iter| iter.iter().cloned())
    }
}

impl<'a, T> Iterator for MultiBufferChunksIter<'a, T> {
    type Item = &'a [T];
    fn next(&mut self) -> Option<Self::Item> {
        self.index = (self.index + 1) % self.data.len();
        self.data
            .get_mut(self.index)
            .as_mut()
            .and_then(|(n, chunks)| chunks.next().as_ref()
                                          .map(|&chunk| if *n == 0 { chunk } else { &chunk[self.npv..] }))
    }
}

#[derive(Debug, Parser)]
pub struct Parameters {
    /// Path to configuration file
    pub config: PathBuf,
    #[arg(long)]
    /// Do not write output to disk
    pub nosave: bool,
    #[arg(long)]
    /// Buffer size in bytes. Defaults to 72MB
    pub buffer_size: Option<usize>,
    #[arg(long)]
    /// Number of threads. Runs in single-threaded mode if not provided
    pub nthreads: Option<usize>
}

impl Parameters {
    pub fn run(self) -> Result<(), MainError> {
        let config_path = self.config;
        let spec = SimulationSpecification::parse(&config_path)?;
        let stride = spec.write_interval.unwrap_or(1);
        let niter = (spec.duration as f64 / spec.time_step).round() as usize / stride * stride;
        let npv = spec.point_vortices.len();
        let npt = spec.passive_tracers.len();
        let n = npv + npt;
        let nthreads = self.nthreads.unwrap_or(1);
        let chunk_size = self.buffer_size.unwrap_or(72 * 1024 * 1024) / 24;
        if npt % nthreads > 0 { Err(error::Error::NThreadsError)? }
        let buf_size_per_thread_per_step = npv + npt / nthreads;
        let buf_size_per_step = buf_size_per_thread_per_step * nthreads;
        let buffer_size = chunk_size / buf_size_per_step * buf_size_per_step;
        let nslice = niter / stride;
        let start_time = Local::now();
        println!("Run started at {}", start_time.format("%m-%d-%Y %H:%M:%S"));

        let mut fbuf = (!self.nosave)
            .then(|| File::create(config_path.with_extension("npy")).map(|f| BufWriter::new(f)))
            .transpose()?;
        let mut writer = fbuf.as_mut()
            .map(|b| npyz::WriteOptions::new()
                .default_dtype()
                .shape(&[nslice as u64, n as u64])
                .writer(b).begin_nd())
            .transpose()?;

        writer.as_mut().map(|w| w.extend(spec.point_vortices.iter().map(|&pv| pv.position))).transpose()?;
        writer.as_mut().map(|w| w.extend(spec.passive_tracers.iter().cloned())).transpose()?;

        if nthreads == 1 {
            let mut mbuf = Vec::new();
            mbuf.reserve_exact(buffer_size);
            let mut time_stepper = kernel::TimeStepper::new(&spec);
            let mut progress = utils::Progress::new(niter);
            for i in 1..niter {
                time_stepper.step();
                if i % stride == 0 {
                    mbuf.extend(time_stepper.state().point_vortices.iter().map(|&pv| pv.position));
                    mbuf.extend(&time_stepper.state().passive_tracers);
                    if mbuf.len() == buffer_size {
                        let data = mbuf.drain(..);
                        writer.as_mut().map(|w| w.extend(data)).transpose()?;
                    }
                }
                progress.step(true);
            }
            writer.as_mut().map(|w| w.extend(mbuf.drain(..))).transpose()?;
        } else {
            let buf_size_per_thread = buffer_size / nthreads;
            let mut mbufs = vec![vec![]; nthreads];
            for mbuf in mbufs.iter_mut() {
                mbuf.reserve_exact(buf_size_per_thread);
            }
            let mut time_steppers = Vec::new();
            for p in spec.divide(nthreads) {
                time_steppers.push(kernel::TimeStepper::new(&p));
            }
            let mut progress = vec![utils::Progress::new(niter); nthreads];
            let buffer_niter = buf_size_per_thread / buf_size_per_thread_per_step * stride;
            let n_segments = (niter + buffer_niter - 2) / buffer_niter;
            for i in 0..n_segments {
                let mut bufs: Vec<_> = mbufs.par_iter_mut()
                    .zip(time_steppers.par_iter_mut())
                    .zip(progress.par_iter_mut())
                    .enumerate()
                    .map(|(n, ((mbuf, time_stepper), progress))| {
                        for _ in ((i == 0) as usize)..buffer_niter {
                            if progress.step >= niter { break }
                            time_stepper.step();
                            if progress.step % stride == 0 {
                                mbuf.extend(time_stepper.state().point_vortices.iter().map(|&pv| pv.position));
                                mbuf.extend(&time_stepper.state().passive_tracers);
                            }
                            progress.step(n == 0);
                        }
                        (n, &mut mbuf[..])
                    })
                    .collect();
                let step_size = npv + npt / nthreads;
                let mut data_stream = MultiBufferData::from(&mut bufs[..], npv, step_size);
                writer.as_mut().map(|w| w.extend(data_stream.iter())).transpose()?;
                for buf in mbufs.iter_mut() { buf.clear(); }
            }
        }
        writer.map(|w| w.finish()).transpose()?;
        Ok(())
    }
}
