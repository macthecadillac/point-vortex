use clap::Parser;
use npyz::WriterBuilder;
use rayon::prelude::*;

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::Path;
use std::slice;

mod config;
mod error;
mod problem;

use main_error::MainError;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to configuration file
    config: String,
    #[arg(long)]
    /// Do not write output to disk
    nosave: bool,
    #[arg(long)]
    /// Buffer size in bytes. Defaults to 72MB
    buffer_size: Option<usize>,
    #[arg(long)]
    /// Number of threads. Runs in single-threaded mode if not provided
    nthreads: Option<usize>
}

#[derive(Clone, Copy)]
struct Progress {
    threshold: f64,
    niter: usize,
    step: usize
}

impl Progress {
    fn new(niter: usize) -> Self {
        print!("0.0% complete\r");
        io::stdout().flush().unwrap();
        Progress { threshold: 0., niter, step: 1 }
    }

    fn step(&mut self, stdout: bool) {
        if stdout {
            let niter_f64 = self.niter as f64;
            let step_f64 = self.step as f64;
            let percent_done = (step_f64 + 1.) * 100. / niter_f64;
            let hundredths = (percent_done * 10.).floor();
            if hundredths > self.threshold {
                self.threshold = hundredths;
                print!("{:.1}% complete\r", percent_done);
                io::stdout().flush().unwrap();
            }
        }
        self.step += 1;
    }
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

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;
    let stride = problem.write_interval.unwrap_or(1);
    let niter = (problem.duration as f64 / problem.time_step).round() as usize / stride * stride;
    let npv = problem.point_vortices.len();
    let npt = problem.passive_tracers.len();
    let n = npv + npt;
    let nthreads = args.nthreads.unwrap_or(1);
    let chunk_size = args.buffer_size.unwrap_or(72 * 1024 * 1024) / 3;
    if npt % nthreads > 0 { Err(error::Error::NThreadsError)? }
    let buf_size_per_thread_per_step = npv + npt / nthreads;
    let buf_size_per_step = buf_size_per_thread_per_step * nthreads;
    let buffer_size = chunk_size / buf_size_per_step * buf_size_per_step;
    let nslice = niter / stride;

    let mut fbuf = (!args.nosave)
        .then(|| File::create(path.with_extension("npy")).map(|f| BufWriter::new(f)))
        .transpose()?;
    let mut writer = fbuf.as_mut()
        .map(|b| npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[nslice as u64, n as u64])
            .writer(b).begin_nd())
        .transpose()?;

    writer.as_mut().map(|w| w.extend(problem.point_vortices.iter().map(|&pv| pv.position))).transpose()?;
    writer.as_mut().map(|w| w.extend(problem.passive_tracers.iter().cloned())).transpose()?;

    if nthreads == 1 {
        let mut mbuf = Vec::new();
        mbuf.reserve_exact(buffer_size);
        let mut solver = problem::Solver::new(&problem);
        let mut progress = Progress::new(niter);
        for i in 1..niter {
            solver.step();
            if i % stride == 0 {
                mbuf.extend(solver.state().point_vortices.iter().map(|&pv| pv.position));
                mbuf.extend(&solver.state().passive_tracers);
            }
            if mbuf.len() == buffer_size {
                let data = mbuf.drain(..);
                writer.as_mut().map(|w| w.extend(data)).transpose()?;
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
        let mut solvers = Vec::new();
        for p in problem.divide(nthreads) {
            solvers.push(problem::Solver::new(&p));
        }
        let mut progress = vec![Progress::new(niter); nthreads];
        let buffer_niter = buf_size_per_thread / buf_size_per_thread_per_step;
        let n_segments = (niter + buffer_niter - 2) / buffer_niter;
        for i in 0..n_segments {
            let mut bufs: Vec<_> = mbufs.par_iter_mut()
                .zip(solvers.par_iter_mut())
                .zip(progress.par_iter_mut())
                .enumerate()
                .map(|(n, ((mbuf, solver), progress))| {
                    for _ in ((i == 0) as usize)..buffer_niter {
                        if progress.step >= niter { break }
                        solver.step();
                        if progress.step % stride == 0 {
                            mbuf.extend(solver.state().point_vortices.iter().map(|&pv| pv.position));
                            mbuf.extend(&solver.state().passive_tracers);
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
