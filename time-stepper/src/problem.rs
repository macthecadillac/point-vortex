use std::f64::consts::FRAC_1_PI;
use std::ops::Mul;

pub trait Problem: Clone + Sized {
    fn sqg(&self) -> bool;
    fn rossby(&self) -> f64;
    fn duration(&self) -> f64;
    fn time_step(&self) -> f64;
    fn point_vortices(&self) -> &[PointVortex];
    fn passive_tracers(&self) -> &[Vector];
    fn replace_tracers(&self, tracers: &[Vector]) -> Self;
    fn divide(&self, n: usize) -> Vec<Self> {
        let pt = self.passive_tracers();
        let chunk_size = (pt.len() + n - 1) / n;
        pt.chunks(chunk_size)
          .map(|chunk| self.replace_tracers(chunk))
          .collect()
    }
}

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

#[derive(Clone)]
pub struct State {
    pub point_vortices: Vec<PointVortex>,
    pub passive_tracers: Vec<Vector>
}

struct Buffer {
    other_vortices: Vec<PointVortex>,
    ks: Vec<[Vector; 4]>,
    state: State
}

pub struct Solver {
    rossby: f64,
    sqg: bool,
    dt: f64,
    state: State,
    buffer: Buffer
}

impl Solver {
    pub fn new(problem: &impl Problem) -> Self {
        let rossby = problem.rossby();
        let dt = problem.time_step();
        let sqg = problem.sqg();
        let state = State {
            point_vortices: problem.point_vortices().to_owned(),
            passive_tracers: problem.passive_tracers().to_owned()
        };
        let n = state.point_vortices.len() + state.passive_tracers.len();
        let buffer = Buffer {
            other_vortices: vec![PointVortex::default(); state.point_vortices.len() - 1],
            ks: vec![[Vector::default(); 4]; n],
            state: state.clone()
        };
        Solver { rossby, sqg, dt, state, buffer }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    pub fn velocity(&self) -> impl Iterator<Item=Vector> + '_ {
        self.buffer.ks.iter().map(|&ks| ks[0])
    }

    fn first_order_change(&mut self, i: usize) {
        let mut ks = self.buffer.ks.iter_mut();
        for ((j, &PointVortex { position, .. }), k) in self.buffer.state.point_vortices.iter()
                                                           .enumerate()
                                                           .zip(&mut ks) {
            self.buffer.other_vortices.clear();
            for &pv in self.buffer.state.point_vortices.iter()
                                   .enumerate()
                                   .filter(|&(k, _)| j != k)
                                   .map(|(_, x)| x) {
                self.buffer.other_vortices.push(pv);
            }
            k[i] = ui(position, &self.buffer.other_vortices, self.rossby, self.sqg);
        }
        for (&pt, k) in self.buffer.state.passive_tracers.iter().zip(&mut ks) {
            k[i] = ui(pt, &self.buffer.state.point_vortices, self.rossby, self.sqg);
        }
    }

    fn euler_est(&mut self, increment: f64, i: usize) {
        self.buffer.state.point_vortices.clear();
        let pvs = &self.state.point_vortices;
        let mut ks = self.buffer.ks.iter_mut();
        for (&PointVortex { position: yn, strength }, k) in pvs.iter().zip(&mut ks) {
            self.buffer.state.point_vortices.push(PointVortex { position: yn + increment * k[i], strength })
        }
        let pts = &self.state.passive_tracers;
        self.buffer.state.passive_tracers.clear();
        for (&yn, k) in pts.iter().zip(&mut ks) {
            self.buffer.state.passive_tracers.push(yn + increment * k[i])
        }
    }

    // Classic Runge-Kutta method
    pub fn step(&mut self) {
        self.first_order_change(0);
        self.euler_est(0.5 * self.dt, 0);
        self.first_order_change(1);
        self.euler_est(0.5 * self.dt, 1);
        self.first_order_change(2);
        self.euler_est(self.dt, 2);
        self.first_order_change(3);
        let c = 1. / 6.;
        let mut ks = self.buffer.ks.iter()
                         .map(|&[k1, k2, k3, k4]| k1 + 2. * k2 + 2. * k3 + k4);
        for ((x_, &x), k) in self.buffer.state.point_vortices
                                 .iter_mut()
                                 .zip(self.state.point_vortices.iter())
                                 .zip(&mut ks) {
            x_.position = x.position + c * self.dt * k
        }
        for ((x_, &x), k) in self.buffer.state.passive_tracers
                                 .iter_mut()
                                 .zip(self.state.passive_tracers.iter())
                                 .zip(&mut ks) {
            *x_ = x + c * self.dt * k
        }
        self.state.point_vortices.copy_from_slice(&self.buffer.state.point_vortices);
        self.state.passive_tracers.copy_from_slice(&self.buffer.state.passive_tracers);
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
    let nvtxs = other_vtxs.len();
    let u1p: Vector = other_vtxs[..nvtxs - 1].iter().enumerate().map(|(i, pv1)| {
            other_vtxs[i + 1..].iter().map(|pv2| {
                let &PointVortex { strength: gamma1, position: vtx1 } = pv1;
                let &PointVortex { strength: gamma2, position: vtx2 } = pv2;
                gamma1 * gamma2 * u1pijk(vtx, vtx1, vtx2, sqg)
            })
        })
        .flatten()
        .sum();
    u0 + rossby * (u1s + u1p)
}
