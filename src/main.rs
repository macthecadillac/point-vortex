extern crate clap;
extern crate derive_more;
extern crate itertools;
extern crate npyz;
extern crate cache_size;
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
        .shape(&[nslice as u64, (npv + npt) as u64])
        .writer(&mut buf)
        .begin_nd()?;
    // write point vortex data, which should be identical across threads
    // (pv_output[0].chunks(npv)
    //     .zip(solver.threads.iter()
    //         .map(|t| t.state().passive_tracers.len())
    //         .zip(pt_output.iter())
    //         .map(|(n, pts)| pts.chunks(n)))  // FIXME: Here lies the problem
    //     .flat_map(|(a, b)| std::iter::once(a).chain(b.into_iter()).flatten()))?;
    let mut ptcs: Vec<_> = solver.threads.iter()
            .map(|t| t.state().passive_tracers.len())
            .zip(pt_output.iter_mut())
            .map(|(n, pts)| pts.chunks_mut(n))
            .collect();
    for pvc in pv_output[0].chunks(npv) {
        writer.extend(pvc.iter())?;
        for ptc in ptcs.iter_mut() {
            let chunk = ptc.next();
            match chunk {
                None => break,
                Some(c) => writer.extend(c.iter())?
            }
        }
    }
    writer.finish()?;
    
    let file = File::create(path.with_extension("npy"))?;
    let mut bufwriter = BufWriter::new(file);
    bufwriter.write(&buf)?;
    Ok(())
}
