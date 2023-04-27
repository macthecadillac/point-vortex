use crate::problem::{PointVortex, Vector};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(serde::Deserialize)]
pub struct Problem {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub point_vortices: Vec<PointVortex>,
    pub passive_tracers: Vec<Vector>,
    pub write_interval: Option<usize>
}

pub fn parse(path: &Path) -> Result<Problem, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
