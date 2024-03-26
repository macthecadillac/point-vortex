use serde::{Deserialize, Deserializer};
use serde::de::Error;

use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

use crate::problem::{PointVortex, Vector};

#[derive(Deserialize)]
#[serde(untagged)]
enum PointOrRange { Point(f64), Range { start: f64, end: f64, n: usize } }

impl TryFrom<PointOrRange> for Vec<f64> {
    type Error = crate::error::Error;
    fn try_from(pr: PointOrRange) -> Result<Vec<f64>, Self::Error> {
        use crate::error::Error;
        match pr {
            PointOrRange::Range { n, .. } if n < 2 => Err(Error::EmptyRange),
            PointOrRange::Range { start, end, .. } if start > end => Err(Error::InvertedRange(start, end)),
            PointOrRange::Range { start, end, n } => {
                let span = end - start;
                let stride = span / (n - 1) as f64;
                Ok((0..n).map(|m| start + m as f64 * stride).collect())
            },
            PointOrRange::Point(p) => Ok(vec![p])
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum GridOrList {
    Grid { xs: PointOrRange, ys: PointOrRange, zs: PointOrRange },
    List(Vec<Vector>)
}

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
    match GridOrList::deserialize(deserializer).map_err(D::Error::custom)? {
        GridOrList::List(l) => Ok(l),
        GridOrList::Grid { xs, ys, zs } => {
            let xpoints: Vec<f64> = xs.try_into().map_err(D::Error::custom)?;
            let ypoints: Vec<f64> = ys.try_into().map_err(D::Error::custom)?;
            let zpoints: Vec<f64> = zs.try_into().map_err(D::Error::custom)?;
            let mut vs = vec![];
            for &x in xpoints.iter() {
                for &y in ypoints.iter() {
                    for &z in zpoints.iter() {
                        vs.push(Vector { x, y, z })
                    }
                }
            }
            Ok(vs)
        }
    }
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
