use crate::config::Problem;

use derive_more::{Add, Sub, Sum, AddAssign};
use itertools::Itertools;
use npyz::WriterBuilder;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;

use std::f64::consts::FRAC_1_PI;
use std::iter::repeat;
use std::num::NonZeroU8;
use std::ops::Mul;

#[derive(Debug, Default, Clone, Copy)]
#[derive(serde::Deserialize)]
#[derive(npyz::AutoSerialize, npyz::Serialize)]
#[derive(Add, Sub, Sum, AddAssign)]
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

struct State {
    pub point_vortices: Vec<PointVortex>,
    pub passive_tracers: Vec<Vector>
}

struct Buffer {
    other_vortices: Vec<PointVortex>,
    ks: [Vec<Vector>; 4],
    yns: [Vec<PointVortex>; 4]
}

struct PassiveTracerTimeStepper {
    rossby: f64,
    sqg: bool,
    dt: f64,
    state: State,
    buffer: Buffer,
}

impl PassiveTracerTimeStepper {
    fn new(point_vortices: &[PointVortex], passive_tracers: &[Vector], rossby: f64,
               dt: f64, sqg: bool) -> Self {
        let state = State {
            point_vortices: point_vortices.to_vec(),
            passive_tracers: passive_tracers.to_vec()
        };
        let n = state.point_vortices.len();
        let buffer = Buffer {
            other_vortices: vec![PointVortex::default(); n - 1],
            ks: [vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n]],
            yns: [vec![PointVortex::default(); n],
                  vec![PointVortex::default(); n],
                  vec![PointVortex::default(); n],
                  vec![PointVortex::default(); n]]
        };
        PassiveTracerTimeStepper { rossby, sqg, dt, state, buffer }
    }

    fn state(&self) -> &State {
        &self.state
    }

    fn first_order_change(&mut self, i: usize) {
        for ((j, k), &PointVortex { position, .. }) in self.buffer.ks[i].iter_mut().enumerate()
                                                           .zip(self.state.point_vortices.iter()) {
            self.buffer.other_vortices.clear();
            for &pv in self.state.point_vortices.iter()
                           .enumerate()
                           .filter(|&(k, _)| j != k)
                           .map(|(_, x)| x) {
                self.buffer.other_vortices.push(pv);
            }
            *k = ui(position, &self.buffer.other_vortices, self.rossby, self.sqg);
        }
    }

    fn euler_est(&mut self, increment: f64, i: usize) {
        self.buffer.yns[i].clear();
        let pvs = &self.state.point_vortices;
        let mut ks = self.buffer.ks[i].iter();
        for (&PointVortex { position: yn, strength }, &k) in pvs.iter().zip(&mut ks) {
            self.buffer.yns[i].push(PointVortex { position: yn + increment * k, strength })
        }
    }

    // Classic Runge-Kutta method
    fn next(&mut self) {
        self.first_order_change(0);
        self.euler_est(0.5 * self.dt, 0);
        self.first_order_change(1);
        self.euler_est(0.5 * self.dt, 1);
        self.first_order_change(2);
        self.euler_est(self.dt, 2);
        self.first_order_change(3);
        for tracer in self.state.passive_tracers.iter_mut() {
            let k1 = ui(*tracer, &self.state.point_vortices, self.rossby, self.sqg);
            let k2 = ui(*tracer + 0.5 * self.dt * k1, &self.buffer.yns[0], self.rossby, self.sqg);
            let k3 = ui(*tracer + 0.5 * self.dt * k2, &self.buffer.yns[1], self.rossby, self.sqg);
            let k4 = ui(*tracer + self.dt * k3, &self.buffer.yns[2], self.rossby, self.sqg);
            *tracer += self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
        }
        for (x, &k1, &k2, &k3, &k4) in self.state.point_vortices.iter_mut()
                                                 .zip(self.buffer.ks[0].iter())
                                                 .zip(self.buffer.ks[1].iter())
                                                 .zip(self.buffer.ks[2].iter())
                                                 .zip(self.buffer.ks[3].iter())
                                                 .map(|((((a, b), c), d), e)| (a, b, c, d, e)) {
            x.position += self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
        }
    }
}

struct Thread {
    time_stepper: PassiveTracerTimeStepper,
    point_vortices: Vec<Vector>,
    passive_tracers: Vec<Vector>
}

pub struct Solver {
    write_interval: usize,
    niter: usize,
    threads: Vec<Thread>,
    npv: usize,
    npt: usize,
    output: Vec<u8>
}

impl Solver {
    pub fn new(problem: &Problem, threads: Option<u8>) -> Self {
        if let Some(n) = threads {
            ThreadPoolBuilder::new().num_threads(n as usize).build_global().unwrap();
        }
        let niter = (problem.duration as f64 / problem.time_step).round() as usize;
        let write_interval = problem.write_interval.unwrap_or(1);
        let nthread = threads.and_then(NonZeroU8::new).map(NonZeroU8::get).unwrap_or(1) as usize;
        let npt = problem.passive_tracers.len();
        let npv = problem.point_vortices.len();
        let threads = problem.passive_tracers.chunks((npt / nthread).max(1)).map(|pt| {
            let pv = &problem.point_vortices;
            let rossby = problem.rossby;
            let dt = problem.time_step;
            let sqg = problem.sqg;
            let time_stepper = PassiveTracerTimeStepper::new(pv, pt, rossby, dt, sqg);
            let pvx = pv.iter().map(|v| v.position)
                               .chain(repeat(Vector::default()))
                               .take(niter / write_interval * pv.len())
                               .collect();
            let ptx = pt.iter().cloned()
                               .chain(repeat(Vector::default()))
                               .take(niter / write_interval * pt.len())
                               .collect();
            Thread { time_stepper, point_vortices: pvx, passive_tracers: ptx }
        }).collect();
        let output = vec![];
        Solver { threads, write_interval, niter, npv, npt, output }
    }

    pub fn solve(&mut self) {
        self.threads.par_iter_mut()
            .for_each(|thread| {
                let nwrites = self.niter / self.write_interval;
                let npt = thread.time_stepper.state().passive_tracers.len();
                let npv = thread.time_stepper.state().point_vortices.len();
                for i in 1..nwrites {
                    for _ in 0..self.write_interval {
                        thread.time_stepper.next();
                    }
                    for (&pv, pv_) in thread.time_stepper.state().point_vortices.iter()
                                                         .zip(thread.point_vortices[i * npv..].iter_mut()) {
                        *pv_ = pv.position
                    }
                    for (&pt, pt_) in thread.time_stepper.state().passive_tracers.iter()
                                                         .zip(thread.passive_tracers[i * npt..].iter_mut()) {
                        *pt_ = pt;
                    }
                }
            })
        ;
    }

    pub fn write(&mut self) -> Result<&[u8], crate::error::Error> {
        let nslice = self.niter / self.write_interval;
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[(self.npv + self.npt) as u64, nslice as u64])
            .writer(&mut self.output)
            .begin_nd()?;
        // write point vortex data, which should be identical across threads
        for i in 0..self.npv {
            writer.extend(self.threads[0].point_vortices.iter().skip(i).step_by(self.npv))?;
        }
        // write tracer data
        for thread in self.threads.iter() {
            let npt = thread.time_stepper.state.passive_tracers.len();
            for i in 0..npt {
                writer.extend(thread.passive_tracers.iter().skip(i).step_by(npt))?;
            }
        }
        writer.finish()?;
        Ok(&self.output)
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
