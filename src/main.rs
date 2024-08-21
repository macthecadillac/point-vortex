use clap::{Parser, Subcommand};
use main_error::MainError;

mod config;
mod error;
mod ftle;
mod poincare_section;
mod kernel;
mod trajectory;
mod utils;

#[derive(Debug, Subcommand)]
enum Subcmd {
    /// Track trajectory of point vortices and passive tracers
    Trajectory(trajectory::Parameters),
    /// Compute finite time Lyapunov exponents
    FTLE(ftle::Parameters),
    /// Compute Poincare sections
    Poincare(poincare_section::Parameters)
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
        Subcmd::Trajectory(t) => t.run()?,
        Subcmd::FTLE(f) => f.run()?,
        Subcmd::Poincare(p) => p.run()?,
    }
    Ok(())
}
