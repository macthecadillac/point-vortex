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

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    config: String,
    #[arg(long)]
    nosave: bool
}

fn main() -> Result<(), main_error::MainError> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;
    let niter = (problem.duration as f64 / problem.time_step).round() as usize;
    let stride = problem.write_interval.unwrap_or(1);
    let n = problem.point_vortices.len() + problem.passive_tracers.len();
    let nslice = niter / stride;

    let mut buf = vec![];
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[nslice as u64, n as u64])
        .writer(&mut buf)
        .begin_nd()?;

    for pv in problem.point_vortices.iter() {
        writer.push(&pv.position)?;
    }

    for tracer in problem.passive_tracers.iter() {
        writer.push(&tracer)?;
    }

    let mut solver = problem::Solver::new(&problem);
    let mut threshold = 1.;
    print!("0.0% complete\r");
    io::stdout().flush().unwrap();
    for i in 1..niter {
        solver.step();
        if i % stride == 0 {
            for pv in solver.state().point_vortices.iter() {
                writer.push(&pv.position)?;
            }
            for pt in solver.state().passive_tracers.iter() {
                writer.push(&pt)?;
            }
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
    writer.finish()?;

    if !args.nosave {
        let file = File::create(path.with_extension("npy"))?;
        let mut writer = BufWriter::new(file);
        writer.write(&buf)?;
    }

    Ok(())
}
