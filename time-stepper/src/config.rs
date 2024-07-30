use serde::{Deserialize, Deserializer};
use serde::de::Error;

use crate::problem::Vector;

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
    pub fn try_into_iter(self) -> Result<RangeIter, crate::error::Error> {
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
pub enum PointOrRange { Point(f64), Range(Range) }

impl PointOrRange {
    pub fn try_into_iter(self) -> Result<RangeIter, crate::error::Error> {
        match self {
            PointOrRange::Range(r) => Ok(r.try_into_iter()?),
            PointOrRange::Point(p) => Ok(RangeIter { start: p, n: 1, step_size: 1. })
        }
    }
}

#[derive(Deserialize)]
pub struct Grid { xs: PointOrRange, ys: PointOrRange, zs: PointOrRange }

impl Grid {
    pub fn try_into_iter(self) -> Result<impl Iterator<Item=Vector>, crate::error::Error> {
        let Grid { xs, ys, zs } = self;
        let xiter = xs.try_into_iter()?;
        let yiter = ys.try_into_iter()?;
        let ziter = zs.try_into_iter()?;
        Ok(xiter.flat_map(move |x| yiter.flat_map(move |y| ziter.map(move |z| Vector { x, y, z }))))
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
pub enum GridOrVector { Grid(Grid), Vector(Vector) }

#[derive(Deserialize)]
pub struct GridOrVectors(Vec<GridOrVector>);

pub fn deserialize_tracers<'de, D>(deserializer: D) -> Result<Vec<Vector>, D::Error>
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
