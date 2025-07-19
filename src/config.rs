use serde::{Deserialize, Deserializer};
use serde::de::{DeserializeOwned, Error};

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::error;
use crate::kernel::{PointVortex, Vector};
use crate::triangle_ps::ScaledDistances;

pub trait Parse: Sized + DeserializeOwned {
    fn parse(path: &Path) -> Result<Self, error::Error> {
        let mut file = File::open(&path)?;
        let mut toml_file = String::new();
        file.read_to_string(&mut toml_file)?;
        let config = toml::from_str(&toml_file)?;
        Ok(config)
    }
}

#[derive(Copy, Clone, Deserialize)]
pub struct Range { start: f64, end: f64, n: usize }

#[derive(Clone, Copy)]
pub struct RangeIter { start: f64, step_size: f64, n: usize }

impl Iterator for RangeIter {
    type Item = f64;
    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            None
        } else {
            let curr = self.start;
            self.start += self.step_size;
            self.n -= 1;
            Some(curr)
        }
    }
}

impl Range {
    pub fn try_into_iter(self) -> Result<RangeIter, error::Error> {
        use error::Error;
        match self {
            Range { n, .. } if n < 2 => Err(Error::EmptyRange),
            Range { start, end, .. } if start > end => Err(Error::InvertedRange(start, end)),
            Range { start, end, n } => {
                let span = end - start;
                let step_size = span / (n - 1) as f64;
                Ok(RangeIter { start, n, step_size })
            },
        }
    }
}

#[derive(Copy, Clone, Deserialize)]
#[serde(untagged)]
pub enum PointOrRange { Point(f64), Range(Range) }

impl PointOrRange {
    pub fn try_into_iter(self) -> Result<RangeIter, error::Error> {
        match self {
            PointOrRange::Range(r) => Ok(r.try_into_iter()?),
            PointOrRange::Point(p) => Ok(RangeIter { start: p, n: 1, step_size: 1. })
        }
    }
}

#[derive(Deserialize)]
pub struct Grid { xs: PointOrRange, ys: PointOrRange, zs: PointOrRange }

impl Grid {
    pub fn try_into_iter(self) -> Result<impl Iterator<Item=Vector>, error::Error> {
        let Grid { xs, ys, zs } = self;
        let xiter = xs.try_into_iter()?;
        let yiter = ys.try_into_iter()?;
        let ziter = zs.try_into_iter()?;
        Ok(xiter.flat_map(move |x| yiter.flat_map(move |y| ziter.map(move |z| Vector { x, y, z }))))
    }
}

pub fn deserialize_grid<'de, D>(deserializer: D) -> Result<Vec<Vector>, D::Error>
    where D: Deserializer<'de> {
    let grid =  Grid::deserialize(deserializer).map_err(D::Error::custom)?;
    let iter = grid.try_into_iter().map_err(D::Error::custom)?;
    Ok(iter.collect())
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum GridOrVector { Grid(Grid), Vector(Vector) }

#[derive(Deserialize)]
pub struct GridOrVectors(Vec<GridOrVector>);

pub fn grid_or_vectors<'de, D>(deserializer: D) -> Result<Vec<Vector>, D::Error>
    where D: Deserializer<'de> {
    let GridOrVectors(pts) =  GridOrVectors::deserialize(deserializer).map_err(D::Error::custom)?;
    let mut tracers = vec![];
    for pt in pts.into_iter() {
        match pt {
            GridOrVector::Vector(v) => tracers.push(v),
            GridOrVector::Grid(grid) => {
                let iter = grid.try_into_iter().map_err(D::Error::custom)?;
                tracers.extend(iter);
            }
        }
    }
    Ok(tracers)
}

#[derive(Deserialize)]
pub struct ScaledDistanceGrid { strength: f64, r12: PointOrRange, r23: PointOrRange }

impl ScaledDistanceGrid {
    pub fn try_into_iter(self) -> Result<impl Iterator<Item=(f64, ScaledDistances)>, error::Error> {
        let ScaledDistanceGrid { strength, r12, r23 } = self;
        let r12iter = r12.try_into_iter()?;
        let r23iter = r23.try_into_iter()?;
        Ok(r12iter.flat_map(move |r12| r23iter.map(move |r23| (strength, ScaledDistances { r12, r23 }))))
    }
}

#[derive(Deserialize)]
pub struct ScaledDistanceRange { min: f64, max: f64, step_size: f64, strength: f64 }

#[derive(Deserialize)]
#[serde(untagged)]
pub enum ScaledDistanceOption {
    Grid(ScaledDistanceGrid), ScaledDistance(f64, ScaledDistances), ScaledDistanceRange(ScaledDistanceRange)
}

#[derive(Deserialize)]
pub struct ScaledDistanceOptions(Vec<ScaledDistanceOption>);

struct PV3 { data: [PointVortex; 3] }

impl TryFrom<(f64, ScaledDistances)> for PV3 {
    type Error = error::Error;
    fn try_from(f: (f64, ScaledDistances)) -> Result<PV3, error::Error> {
        let (strength, ScaledDistances { r12, r23 }) = f;
        let v1 = PointVortex { strength, position: Vector { x: 0., y: 0., z: 0. } };
        let v3 = PointVortex { strength, position: Vector { x: 1., y: 0., z: 0. } };
        let cos_theta12 = 0.5 * (1. - r23 * r23 + r12 * r12) / r12;
        let x = r12 * cos_theta12;
        if cos_theta12 > 1. {
            Err(error::Error::TriangleInequalityError(r12, r23))
        } else {
            let sin2 = 1. - cos_theta12 * cos_theta12;
            let y = r12 * sin2.sqrt();
            let v2 = PointVortex { strength, position: Vector { x, y, z: 0. } };
            Ok(PV3 { data: [v1, v2, v3] })
        }
    }
}

pub fn grid_or_distances<'de, D>(deserializer: D) -> Result<Vec<[PointVortex; 3]>, D::Error>
    where D: Deserializer<'de> {
    let ScaledDistanceOptions(pvs) =
        ScaledDistanceOptions::deserialize(deserializer).map_err(D::Error::custom)?;
    let mut active_tracers = vec![];
    for pv in pvs.into_iter() {
        match pv {
            ScaledDistanceOption::ScaledDistanceRange(r) => {
                let ScaledDistanceRange { min, max, step_size, strength } = r;
                let start = min;
                let end = max;
                let n = ((end - start) / step_size).floor() as usize;
                let r12range = Range { start, end, n };
                let iter = r12range.try_into_iter()
                    .map_err(D::Error::custom)?
                    .flat_map(|r12| {
                        let start = (1. - r12 + 1e-8).abs().max(r12);
                        let end = 1. + r12;
                        let n = ((end - start) / step_size).floor() as usize;
                        let r23range = RangeIter { start, step_size, n };
                        r23range.map(move |r23| (strength, ScaledDistances { r12, r23 }))
                    })
                    .map(PV3::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(D::Error::custom)?
                    .into_iter()
                    .map(|pv| pv.data);
                active_tracers.extend(iter);
            }
            ScaledDistanceOption::ScaledDistance(strength, v) => {
                let pv = PV3::try_from((strength, v)).map_err(D::Error::custom)?.data;
                active_tracers.push(pv)
            },
            ScaledDistanceOption::Grid(grid) => {
                let iter = grid.try_into_iter().map_err(D::Error::custom)?
                    .map(PV3::try_from)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(D::Error::custom)?
                    .into_iter()
                    .map(|pv| pv.data);
                active_tracers.extend(iter);
            }
        }
    }
    Ok(active_tracers)
}
