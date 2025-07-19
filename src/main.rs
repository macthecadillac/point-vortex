use clap::{Parser, Subcommand};
use main_error::MainError;

mod config;
mod error;
mod excursion;
mod ftle;
mod rsps;
mod triangle_ps;
mod triangle_side_lengths;
mod kernel;
mod tracker;
mod utils;

#[derive(Debug, Subcommand)]
enum Subcmd {
    /// Track trajectory of point vortices and passive tracers
    Track(tracker::Parameters),
    /// Compute finite time Lyapunov exponents
    FTLE(ftle::Parameters),
    /// Compute Poincare sections in real space
    RealSpacePoincareSection(rsps::Parameters),
    /// Compute Poincare sections in real space
    TrianglePoincareSection(triangle_ps::Parameters),
    /// Track triangle side lengths
    TriangleSideLengths(triangle_side_lengths::Parameters),
    /// Find the amount of vertical movement of tracers started at a given position
    VerticalExcursion(excursion::Parameters)
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
        Subcmd::RealSpacePoincareSection(p) => p.run()?,
        Subcmd::TrianglePoincareSection(p) => p.run()?,
        Subcmd::TriangleSideLengths(p) => p.run()?,
        Subcmd::VerticalExcursion(p) => p.run()?
    }
    Ok(())
}
