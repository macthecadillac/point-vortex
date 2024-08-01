use serde::Deserialize;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use time_stepper::problem::{PointVortex, Vector};

#[derive(Deserialize)]
#[derive(Clone)]
pub struct Problem {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "time_stepper::config::grid_or_vectors")]
    pub passive_tracers: Vec<Vector>,
    pub write_interval: Option<usize>,
}

impl time_stepper::problem::Problem for Problem {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn duration(&self) -> f64 { self.duration }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn passive_tracers(&self) -> &[Vector] { &self.passive_tracers }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self { Self { passive_tracers: tracers.to_owned(), ..self.clone() } }
}

pub fn parse(path: &Path) -> Result<Problem, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
