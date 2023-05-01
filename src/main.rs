extern crate clap;
extern crate derive_more;
extern crate itertools;
extern crate npyz;
extern crate serde;
extern crate toml;

use clap::Parser;
use npyz::WriterBuilder;

use std::fs::File;
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
    threads: Option<u8>
}

fn main() -> Result<(), error::Error> {
    let args = Args::parse();
    let path = Path::new(&args.config);
    let problem = config::parse(&path)?;

    let mut solver = problem::Solver::new(&problem, args.threads);

    let niter = (problem.duration as f64 / problem.time_step).round() as usize;
    let write_interval = problem.write_interval.unwrap_or(1);
    let nslice = niter / write_interval;
    let npv = problem.point_vortices.len();
    let npt = problem.passive_tracers.len();
    let mut buf = vec![];
    
    let (mut pv_output, mut pt_output) = solver.create_buffer();
    solver.solve(&mut pv_output, &mut pt_output);
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(&[(npv + npt) as u64, nslice as u64])
        .writer(&mut buf)
        .begin_nd()?;
    // write point vortex data, which should be identical across threads
    for i in 0..npv {
        writer.extend(pv_output[0].iter().skip(i).step_by(npv))?;
    }
    for (thread, pts) in solver.threads.iter().zip(pt_output.iter()) {
        let passive_tracers = &thread.state().passive_tracers;
        let npt = passive_tracers.len();
        for j in 0..npt {
            writer.extend(pts.iter().skip(j).step_by(npt))?;
        }
    }
    writer.finish()?;
    
    let file = File::create(path.with_extension("npy"))?;
    let mut bufwriter = BufWriter::new(file);
    bufwriter.write(&buf)?;
    Ok(())
}
