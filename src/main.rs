use clap::{Parser, Subcommand};
use main_error::MainError;

mod config;
///// complexity measure
//mod cm;
mod error;
mod ftle;
mod poincare_section;
mod problem;
mod trajectory;

#[derive(Debug, Subcommand)]
enum Subcmd {
    /// Track trajectory of point vortices and passive tracers
    Trajectory(trajectory::Parameters),
    /// Compute finite time Lyapunov exponents
    FTLE(ftle::Parameters),
    /// Compute Poincare sections
    Poincare(poincare_section::Parameters)
    ///// Compute complexity measure
    //CM(cm::Parameters)
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
        //Subcmd::CM(d) => d.run()?
    }
    Ok(())
}
