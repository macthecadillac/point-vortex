use serde::{Deserialize, Deserializer};
use serde::de::Error;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::problem::{PointVortex, Vector};

#[derive(Copy, Clone, Deserialize)]
struct Range { start: f64, end: f64, n: usize }

#[derive(Clone, Copy)]
struct RangeIter { start: f64, step_size: f64, n: usize }

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
    fn try_into_iter(self) -> Result<RangeIter, crate::error::Error> {
        use crate::error::Error;
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
enum PointOrRange { Point(f64), Range(Range) }

impl PointOrRange {
    fn try_into_iter(self) -> Result<RangeIter, crate::error::Error> {
        match self {
            PointOrRange::Range(r) => Ok(r.try_into_iter()?),
            PointOrRange::Point(p) => Ok(RangeIter { start: p, n: 1, step_size: 1. })
        }
    }
}

#[derive(Deserialize)]
struct Grid { xs: PointOrRange, ys: PointOrRange, zs: PointOrRange }

impl Grid {
    fn try_into_iter(self) -> Result<impl Iterator<Item=Vector>, crate::error::Error> {
        let Grid { xs, ys, zs } = self;
        let xiter = xs.try_into_iter()?;
        let yiter = ys.try_into_iter()?;
        let ziter = zs.try_into_iter()?;
        Ok(xiter.flat_map(move |x| yiter.flat_map(move |y| ziter.map(move |z| Vector { x, y, z }))))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GridOrVector { Grid(Grid), Vector(Vector) }

#[derive(Deserialize)]
struct GridOrVectors(Vec<GridOrVector>);

#[derive(Deserialize)]
#[derive(Clone)]
pub(crate) struct Problem {
    pub(crate) sqg: bool,
    pub(crate) rossby: f64,
    pub(crate) duration: f64,
    pub(crate) time_step: f64,
    pub(crate) point_vortices: Vec<PointVortex>,
    #[serde(deserialize_with = "deserialize_tracers")]
    pub(crate) passive_tracers: Vec<Vector>,
    pub(crate) write_interval: Option<usize>,
}

fn deserialize_tracers<'de, D>(deserializer: D) -> Result<Vec<Vector>, D::Error>
    where D: Deserializer<'de>, {
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

impl Problem {
    pub(crate) fn divide(&self, n: usize) -> impl Iterator<Item=Self> + '_ {
        let npt = self.passive_tracers.len();
        let chunk_size = (npt + n - 1) / n;
        self.passive_tracers.chunks(chunk_size)
            .map(|chunk| Self { passive_tracers: chunk.to_owned(), ..self.clone() })
    }
}

pub(crate) fn parse(path: &Path) -> Result<Problem, crate::error::Error> {
    let mut file = File::open(&path)?;
    let mut toml_file = String::new();
    file.read_to_string(&mut toml_file)?;

    let config = toml::from_str(&toml_file)?;
    Ok(config)
}
