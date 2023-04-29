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

#[derive(serde::Deserialize, Copy, Clone, Default)]
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
    ks: [Vec<Vector>; 4]
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
        let buffer = Buffer {
            other_vortices: vec![PointVortex::default(); state.point_vortices.len() - 1],
            ks: [vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n]]
        };
        if let Some(n) = threads {
            ThreadPoolBuilder::new().num_threads(n as usize).build_global().unwrap();
        }
        Solver { rossby, sqg, dt, state, buffer, threads }
    }

    fn first_order_change(&mut self, i: usize, state: &State) {
        self.buffer.ks[i].clear();
        for (j, &PointVortex { position, .. }) in state.point_vortices.iter().enumerate() {
            self.buffer.other_vortices.clear();
            for &pv in state.point_vortices.iter()
                                   .enumerate()
                                   .filter(|&(k, _)| j != k)
                                   .map(|(_, x)| x) {
                self.buffer.other_vortices.push(pv);
            }
            self.buffer.ks[i].push(ui(position, &self.buffer.other_vortices, self.rossby, self.sqg));
        }
        // if self.threads.is_some() {
        //     let tmp: Vec<_> = state.passive_tracers.par_iter().map(|&pt| {
        //         ui(pt, &state.point_vortices, self.rossby, self.sqg)
        //     })
        //     .collect();
        //     self.buffer.ks[i].extend_from_slice(&tmp);
        // }
        // else {
        //     for &pt in state.passive_tracers.iter() {
        //         self.buffer.ks[i].push(ui(pt, &state.point_vortices, self.rossby, self.sqg));
        //     }
        // }
    }

    fn euler_est(&self, slope: &[Vector], increment: f64, output: &mut State) {
        output.point_vortices.clear();
        let pvs = &self.state.point_vortices;
        let mut ks = slope.iter();
        for (&PointVortex { position: yn, strength }, &k) in pvs.iter().zip(&mut ks) {
            output.point_vortices.push(PointVortex { position: yn + increment * k, strength })
        }
        // let pts = &self.state.passive_tracers;
        // output.passive_tracers.clear();
        // for (&yn, &k) in pts.iter().zip(&mut ks) {
        //     output.passive_tracers.push(yn + increment * k)
        // }
    }

    // Classic Runge-Kutta method
    pub fn step(&mut self, buffer: &mut State) {
        self.first_order_change(0, &buffer);
        self.euler_est(&self.buffer.ks[0], 0.5 * self.dt, buffer);
        self.first_order_change(1, &buffer);
        self.euler_est(&self.buffer.ks[1], 0.5 * self.dt, buffer);
        self.first_order_change(2, &buffer);
        self.euler_est(&self.buffer.ks[2], self.dt, buffer);
        self.first_order_change(3, &buffer);
        let c = 1. / 6.;
        let mut k1s_iter = self.buffer.ks[0].iter();
        let mut k2s_iter = self.buffer.ks[1].iter();
        let mut k3s_iter = self.buffer.ks[2].iter();
        let mut k4s_iter = self.buffer.ks[3].iter();
        for (((((x_, &x), &k1), &k2), &k3), &k4) in buffer.point_vortices
                                                          .iter_mut()
                                                          .zip(self.state.point_vortices.iter())
                                                          .zip(&mut k1s_iter)
                                                          .zip(&mut k2s_iter)
                                                          .zip(&mut k3s_iter)
                                                          .zip(&mut k4s_iter) {
            x_.position = x.position + c * self.dt * (k1 + 2. * k2 + 2. * k3 + k4)
        }
        for (((((x_, &x), &k1), &k2), &k3), &k4) in buffer.passive_tracers
                                                          .iter_mut()
                                                          .zip(self.state.passive_tracers.iter())
                                                          .zip(&mut k1s_iter)
                                                          .zip(&mut k2s_iter)
                                                          .zip(&mut k3s_iter)
                                                          .zip(&mut k4s_iter) {
            *x_ = x + c * self.dt * (k1 + 2. * k2 + 2. * k3 + k4)
        }
        self.state.point_vortices.copy_from_slice(&buffer.point_vortices);
        self.state.passive_tracers.copy_from_slice(&buffer.passive_tracers);
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
