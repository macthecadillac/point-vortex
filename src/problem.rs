use crate::config::Problem;

use itertools::Itertools;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::f64::consts::FRAC_1_PI;
use std::ops::Mul;

#[derive(Debug, Default, Clone, Copy)]
#[derive(serde::Deserialize)]
#[derive(npyz::AutoSerialize, npyz::Serialize)]
#[derive(derive_more::Add, derive_more::Sub, derive_more::Sum)]
pub struct Vector { pub x: f64, pub y: f64, pub z: f64 }

impl Mul<Vector> for f64 {
    type Output = Vector;
    fn mul(self, other: Vector) -> Self::Output {
        Vector { x: self * other.x,  y: self * other.y,  z: self * other.z }
    }
}

impl Vector {
    fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn norm_pow(self, pow: f64) -> f64 {
        self.norm_sq().powf(0.5 * pow)
    }
}

#[derive(serde::Deserialize, Copy, Clone)]
#[derive(Default)]
pub struct PointVortex {
    pub strength: f64,
    pub position: Vector
}

pub struct State {
    pub point_vortices: Vec<PointVortex>,
    pub passive_tracers: Vec<Vector>
}

struct Buffer {
    other_vortices: Vec<PointVortex>,
    ks: [Vec<Vector>; 4],
    pv_yns: [Vec<PointVortex>; 4]
}

pub struct Solver {
    rossby: f64,
    sqg: bool,
    dt: f64,
    state: State,
    buffer: Buffer,
    threads: Option<u8>
}

impl Solver {
    pub fn new(problem: &Problem, threads: Option<u8>) -> Self {
        let rossby = problem.rossby;
        let dt = problem.time_step;
        let sqg = problem.sqg;
        let state = State {
            point_vortices: problem.point_vortices.clone(),
            passive_tracers: problem.passive_tracers.clone()
        };
        let n = state.point_vortices.len() + state.passive_tracers.len();
        if let Some(n) = threads {
            ThreadPoolBuilder::new().num_threads(n as usize).build_global().unwrap();
        }
        let buffer = Buffer {
            other_vortices: vec![PointVortex::default(); n - 1],
            ks: [vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n]],
            pv_yns: [vec![PointVortex::default(); n],
                     vec![PointVortex::default(); n],
                     vec![PointVortex::default(); n],
                     vec![PointVortex::default(); n]]
        };
        Solver { rossby, sqg, dt, state, threads, buffer }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn step(&mut self) {
        for (ks, yns, c) in self.buffer.ks.iter_mut()
                                          .zip(self.buffer.pv_yns.iter_mut())
                                          .zip([0.5, 0.5, 1.0, 1.0])
                                          .map(|((a, b), c)| (a, b, c)) {
            for (j, yn, &PointVortex { position: y0, .. }, k) in yns.iter_mut()
                                                                    .enumerate()
                                                                    .zip(self.state.point_vortices.iter())
                                                                    .zip(ks.iter_mut())
                                                                    .map(|(((a, b), c), d)| (a, b, c, d)) {
                for (&pv, opv) in self.state.point_vortices.iter()
                                                       .enumerate()
                                                       .filter(|&(k, _)| j != k)
                                                       .map(|(_, x)| x)
                                                       .zip(self.buffer.other_vortices.iter_mut()) {
                    *opv = pv;
                }
                *k = ui(y0, &self.buffer.other_vortices, self.rossby, self.sqg);
                yn.position = y0 + c * self.dt * (*k);
            }
        }
        let update = |y0: &mut Vector| {
            let k1 = ui(*y0, &self.state.point_vortices, self.rossby, self.sqg);
            let k2 = ui(*y0, &self.buffer.pv_yns[0], self.rossby, self.sqg);
            let k3 = ui(*y0, &self.buffer.pv_yns[1], self.rossby, self.sqg);
            let k4 = ui(*y0, &self.buffer.pv_yns[2], self.rossby, self.sqg);
            *y0 = *y0 + 1. / 6. * self.dt * (k1 + 2. * k2 + 2. * k3 + k4)
        };
        match self.threads {
            Some(_) => self.state.passive_tracers.par_iter_mut().for_each(update),
            None => self.state.passive_tracers.iter_mut().for_each(update)
        };
        for (y0, &k1, &k2, &k3, &k4) in self.state.point_vortices.iter_mut()
                                                                 .zip(self.buffer.ks[0].iter())
                                                                 .zip(self.buffer.ks[1].iter())
                                                                 .zip(self.buffer.ks[2].iter())
                                                                 .zip(self.buffer.ks[3].iter())
                                                                 .map(|((((a, b), c), d), e)| (a, b, c, d, e)) {
            let position = y0.position + 1./ 6. * self.dt * (k1 + 2. * k2 + 2. * k3 + k4);
            *y0 = PointVortex { position, strength: y0.strength };
        }
    }
}

fn u0ij(vtx: Vector, vtx1: Vector, sqg: bool) -> Vector {
    let c = if sqg { 0.5 } else { 0.25 };
    let dx = vtx - vtx1;
    let coeff = c * FRAC_1_PI * dx.norm_pow(-3.);
    Vector { x: -dx.y * coeff, y: dx.x * coeff, z: 0. }
}

fn u1sij(vtx: Vector, vtx1: Vector, sqg: bool) -> Vector {
    let c = if sqg { 0.25 } else { 0.0625 };
    let dx = vtx - vtx1;
    let csum = dx.x * dx.x + dx.y * dx.y - 8. * dx.z * dx.z;
    let coeff = c * FRAC_1_PI * FRAC_1_PI * dx.norm_pow(-8.) * csum;
    Vector { x: -dx.y * coeff , y: dx.x * coeff, z: 0. }
}

fn u1pijk(vtx: Vector, vtx1: Vector, vtx2: Vector, sqg: bool) -> Vector {
    let c = if sqg { 0.25 } else { 0.0625 };
    let dx1 = vtx - vtx1;
    let dx2 = vtx - vtx2;
    let Vector { x: x1, y: y1, z: z1 } = dx1;
    let Vector { x: x2, y: y2, z: z2 } = dx2;
    let coeff = c * FRAC_1_PI * FRAC_1_PI * dx1.norm_pow(-5.) * dx2.norm_pow(-5.);
    let dx1_sq = dx1.norm_sq();
    let dx2_sq = dx2.norm_sq();
    let x =
        3. * dx1_sq * (y1 * z2 * z2 + 2. * y2 * z1 * z2) +
        3. * dx2_sq * (y2 * z1 * z1 + 2. * y1 * z1 * z2) -
        dx1_sq * dx2_sq * (y1 + y2);
    let y =
        dx1_sq * dx2_sq * (x1 + x2) -
        3. * dx1_sq * (x1 * z2 * z2 + 2. * x2 * z1 * z2) -
        3. * dx2_sq * (x2 * z1 * z1 + 2. * x1 * z1 * z2);
    let z = 3. * (x2 * y1 - x1 * y2) * (dx2_sq * z1 - dx1_sq * z2);
    coeff * Vector { x, y, z }
}

fn ui(vtx: Vector, other_vtxs: &[PointVortex], rossby: f64, sqg: bool) -> Vector {
    let u0: Vector = other_vtxs
        .iter()
        .map(|&pv| pv.strength * u0ij(vtx, pv.position, sqg))
        .sum();
    let u1s: Vector = other_vtxs
        .iter()
        .map(|&pv| pv.strength * pv.strength * u1sij(vtx, pv.position, sqg))
        .sum();
    let u1p: Vector = other_vtxs
        .iter()
        .combinations(2)
        .map(|pvs| {
            let &PointVortex { strength: gamma1, position: vtx1 } = pvs[0];
            let &PointVortex { strength: gamma2, position: vtx2 } = pvs[1];
            gamma1 * gamma2 * u1pijk(vtx, vtx1, vtx2, sqg)
        })
        .sum();
    u0 + rossby * (u1s + u1p)
}
