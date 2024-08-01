use serde::Deserialize;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use time_stepper::problem::{PointVortex, Problem, Vector};

#[derive(Deserialize)]
#[derive(Clone)]
pub struct P {
    pub sqg: bool,
    pub rossby: f64,
    pub time_step: f64,
    pub delta: f64,
    pub tol: f64,
    pub point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "time_stepper::config::deserialize_grid")]
    pub grid_points: Vec<Vector>
}

impl P {
    pub(crate) fn add_delta_tracers(&self) -> Self {
        let mut grid_points = vec![];
        for &v in self.grid_points.iter() {
            grid_points.push(v);
            grid_points.push(Vector { x: v.x + self.delta, ..v });
            grid_points.push(Vector { y: v.y + self.delta, ..v });
            grid_points.push(Vector { z: v.z - self.delta, ..v });
        }
        Self { grid_points, ..self.clone() }
    }
}

impl Problem for P {
    fn sqg(&self) -> bool { self.sqg }
    fn rossby(&self) -> f64 { self.rossby }
    fn duration(&self) -> f64 { 0. }
    fn time_step(&self) -> f64 { self.time_step }
    fn point_vortices(&self) -> &[PointVortex] { &self.point_vortices }
    fn replace_tracers(&self, tracers: &[Vector]) -> Self {
        Self {
            sqg: self.sqg,
            rossby: self.rossby,
            time_step: self.time_step,
            delta: self.delta,
            tol: self.tol,
            grid_points: tracers.to_owned(),
            point_vortices: self.point_vortices.clone(),
        }
    }
    fn passive_tracers(&self) -> &[Vector] { &self.grid_points }
}

pub fn parse(path: &Path) -> Result<P, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
