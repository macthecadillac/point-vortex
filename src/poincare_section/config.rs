use serde::Deserialize;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::problem::{PointVortex, Vector};

#[derive(Deserialize)]
#[derive(Copy, Clone, Debug)]
// a x + b y + c z + d = 0
pub struct Plane { a: f64, b: f64, c: f64, d: f64 }

impl Plane {
    fn dist(self, v: Vector) -> f64 {
        self.a * v.x + self.b * v.y + self.c * v.z + self.d
    }

    pub fn section(self, prev: Vector, next: Vector) -> Option<Vector> {
        let p = self.dist(prev);
        let n = self.dist(next);
        if p > 0. && n < 0. {
            Some((p * next - n * prev) / (p - n))
        } else {
            None
        }
    }
}

#[derive(Deserialize)]
#[derive(Clone)]
pub struct Problem {
    pub sqg: bool,
    pub rossby: f64,
    pub duration: f64,
    pub time_step: f64,
    pub plane: Plane,
    pub point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "crate::config::grid_or_vectors")]
    pub passive_tracers: Vec<Vector>
}

impl crate::problem::Problem for Problem {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
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
