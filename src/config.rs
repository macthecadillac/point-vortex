use crate::problem::{PointVortex, Vector};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(serde::Deserialize)]
#[derive(Clone)]
pub struct Problem {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub point_vortices: Vec<PointVortex>,
    pub passive_tracers: Vec<Vector>,
    pub write_interval: Option<usize>,
}

impl Problem {
    pub fn divide(&self, n: usize) -> impl Iterator<Item=Self> + '_ {
        let npt = self.passive_tracers.len();
        let chunk_size = (npt + n - 1) / n;
        self.passive_tracers.chunks(chunk_size)
            .map(|chunk| {
                Self { passive_tracers: chunk.to_owned(),
                       ..self.clone() }
            })
    }

    pub fn npt(&self) -> usize { self.passive_tracers.len() }
    pub fn npv(&self) -> usize { self.point_vortices.len() }
}

pub fn parse(path: &Path) -> Result<Problem, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
