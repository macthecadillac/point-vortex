use crate::problem::{PointVortex, Vector};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

#[derive(serde::Deserialize)]
#[derive(Clone)]
pub(crate) struct Problem {
    pub(crate) sqg: bool,
    pub(crate) rossby: f64,
    pub(crate) duration: f64,
    pub(crate) time_step: f64,
    pub(crate) point_vortices: Vec<PointVortex>,
    pub(crate) passive_tracers: Vec<Vector>,
    pub(crate) write_interval: Option<usize>,
}

impl Problem {
    pub(crate) fn divide(&self, n: usize) -> impl Iterator<Item=Self> + '_ {
        let npt = self.passive_tracers.len();
        let chunk_size = (npt + n - 1) / n;
        self.passive_tracers.chunks(chunk_size)
            .map(|chunk| {
                Self { passive_tracers: chunk.to_owned(),
                       ..self.clone() }
            })
    }
}

pub(crate) fn parse(path: &Path) -> Result<Problem, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
