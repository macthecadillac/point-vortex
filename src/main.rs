use clap::Parser;
use npyz::WriterBuilder;
use rayon::prelude::*;

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::Path;

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
        Progress { threshold: 0., niter, step: 0 }
    }

    fn report(&mut self) {
        self.step += 1;
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
}

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;
    let niter = (problem.duration as f64 / problem.time_step).round() as usize;
    let stride = problem.write_interval.unwrap_or(1);
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
        // let mut threshold = 0.;
        let mut progress = Progress::new(niter);
        print!("0.0% complete\r");
        io::stdout().flush().unwrap();
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
            progress.report();
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
                    for step in ((i == 0) as usize)..buffer_niter {
                        let total_steps = i * buffer_niter + step;
                        if total_steps >= niter { break }
                        solver.step();
                        if n == 0 { progress.report() }
                        if total_steps % stride == 0 {
                            mbuf.extend(solver.state().point_vortices.iter().map(|&pv| pv.position));
                            mbuf.extend(&solver.state().passive_tracers);
                        }
                    }
                    (n, mbuf)
                })
                .collect();
            bufs.sort_by(|(a, _), (b, _)| a.cmp(b));
            let step_size = npv + npt / nthreads;
            let mut chunked: Vec<_> = bufs.iter_mut().map(|(n, buf)| (*n, buf.chunks(step_size))).collect();
            loop {
                let mut br = false;
                for (n, ref mut chunks) in chunked.iter_mut() {
                    if let Some(chunk) = chunks.next() {
                        if *n == 0 {
                            writer.as_mut().map(|w| w.extend(chunk.iter().cloned())).transpose()?;
                        } else {
                            writer.as_mut().map(|w| w.extend(chunk[npv..].iter().cloned())).transpose()?;
                        }
                    } else {
                        br = true;
                        break
                    }
                }
                if br { break }
            };
            for (_, buf) in bufs.iter_mut() { buf.clear(); }
        }
    }
    writer.map(|w| w.finish()).transpose()?;
    Ok(())
}
