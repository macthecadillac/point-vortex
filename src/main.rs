extern crate clap;
extern crate derive_more;
extern crate itertools;
extern crate npyz;
extern crate serde;
extern crate toml;

use clap::Parser;

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
    let mut output = solver.create_buffer();
    solver.solve(&mut output);
    let buf = output.write()?;
    
    let file = File::create(path.with_extension("npy"))?;
    let mut bufwriter = BufWriter::new(file);
    bufwriter.write(&buf)?;
    Ok(())
}
