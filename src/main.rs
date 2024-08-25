use clap::{Parser, Subcommand};
use main_error::MainError;

mod config;
mod error;
mod ftle;
mod poincare_section;
mod kernel;
mod tracker;
mod utils;

#[derive(Debug, Subcommand)]
enum Subcmd {
    /// Track trajectory of point vortices and passive tracers
    Track(tracker::Parameters),
    /// Compute finite time Lyapunov exponents
    FTLE(ftle::Parameters),
    /// Compute Poincare sections
    RealSpacePoincare(poincare_section::Parameters)
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[command(subcommand)]
    command: Subcmd,
}

fn main() -> Result<(), MainError> {
    let args = Args::parse();
    match args.command {
        Subcmd::Track(t) => t.run()?,
        Subcmd::FTLE(f) => f.run()?,
        Subcmd::RealSpacePoincare(p) => p.run()?,
    }
    Ok(())
}
