use clap::Parser;
use npyz::WriterBuilder;

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
    config: String,
    #[arg(long)]
    nosave: bool
}

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;
    let niter = (problem.duration as f64 / problem.time_step).round() as usize;
    let stride = problem.write_interval.unwrap_or(1);
    let n = problem.point_vortices.len() + problem.passive_tracers.len();
    let chunk_size = problem.chunk_size.unwrap_or(1024 * 1024);
    let buffer_size = chunk_size / n * n;
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

    let mut mbuf = Vec::new();
    mbuf.reserve_exact(buffer_size);

    let mut solver = problem::Solver::new(&problem);
    let mut threshold = 0.;
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
        let niter_f64 = niter as f64;
        let i_f64 = i as f64;
        let percent_done = (i_f64 + 1.) * 100. / niter_f64;
        let hundredths = (percent_done * 10.).floor();
        if hundredths > threshold {
            threshold = hundredths;
            print!("{:.1}% complete\r", percent_done);
            io::stdout().flush().unwrap();
        }
    }
    writer.as_mut().map(|w| w.extend(mbuf.drain(..))).transpose()?;
    writer.map(|w| w.finish()).transpose()?;
    Ok(())
}
