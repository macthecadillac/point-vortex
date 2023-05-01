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

#[derive(Copy, Clone, Default)]
pub struct IndexedPointVortex {
    pub index: usize,
    pub point_vortex: PointVortex
}

#[derive(Copy, Clone, Default)]
pub struct IndexedPassiveTracer {
    pub index: usize,
    pub position: Vector
}

#[derive(Copy, Clone, Default)]
pub enum Data {
    #[default]
    Empty,
    PointVortex(IndexedPointVortex),
    PassiveTracer(IndexedPassiveTracer)
}

pub struct State {
    pub point_vortices: Vec<IndexedPointVortex>,
    pub passive_tracers: Vec<IndexedPassiveTracer>
}

struct Buffer {
    other_vortices: Vec<IndexedPointVortex>,
    ks: [Vec<Vector>; 4],
    yns: [Vec<IndexedPointVortex>; 4]
}

#[derive(Copy, Clone)]
enum ObjectIndex {
    PointVortex(u8),
    PassiveTracer(u8)
}

pub struct TimeStepper {
    rossby: f64,
    sqg: bool,
    dt: f64,
    state: State,
    buffer: Buffer,
    output_queue: Vec<ObjectIndex>
}

impl TimeStepper {
    fn new(point_vortices: &[IndexedPointVortex],
           passive_tracers: &[IndexedPassiveTracer],
           rossby: f64, dt: f64, sqg: bool) -> Self {
        let state = State {
            point_vortices: point_vortices.to_vec(),
            passive_tracers: passive_tracers.to_vec()
        };
        let n = state.point_vortices.len();
        let buffer = Buffer {
            other_vortices: vec![IndexedPointVortex::default(); n - 1],
            ks: [vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n],
                 vec![Vector::default(); n]],
            yns: [vec![IndexedPointVortex::default(); n],
                  vec![IndexedPointVortex::default(); n],
                  vec![IndexedPointVortex::default(); n],
                  vec![IndexedPointVortex::default(); n]]
        };
        let output_queue = vec![];
        TimeStepper { rossby, sqg, dt, state, buffer, output_queue }
    }

    pub fn state(&self) -> &State {
        &self.state
    }

    fn first_order_change(&mut self, i: usize) {
        for ((j, k), &ipv) in self.buffer.ks[i].iter_mut()
                                  .enumerate()
                                  .zip(self.state.point_vortices.iter()) {
            self.buffer.other_vortices.clear();
            for &pv in self.state.point_vortices.iter()
                           .enumerate()
                           .filter(|&(k, _)| j != k)
                           .map(|(_, x)| x) {
                self.buffer.other_vortices.push(pv);
            }
            *k = ui(ipv.point_vortex.position, &self.buffer.other_vortices,
                    self.rossby, self.sqg);
        }
    }

    fn euler_est(&mut self, increment: f64, i: usize) {
        self.buffer.yns[i].clear();
        let pvs = &self.state.point_vortices;
        let mut ks = self.buffer.ks[i].iter();
        for (&ipv, &k) in pvs.iter().zip(&mut ks) {
            let yn = ipv.point_vortex.position;
            let point_vortex = PointVortex { position: yn + increment * k,
                                             ..ipv.point_vortex };
            self.buffer.yns[i].push(IndexedPointVortex { point_vortex, ..ipv })
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
            let k1 = ui(tracer.position, &self.state.point_vortices, self.rossby, self.sqg);
            let k2 = ui(tracer.position + 0.5 * self.dt * k1, &self.buffer.yns[0], self.rossby, self.sqg);
            let k3 = ui(tracer.position + 0.5 * self.dt * k2, &self.buffer.yns[1], self.rossby, self.sqg);
            let k4 = ui(tracer.position + self.dt * k3, &self.buffer.yns[2], self.rossby, self.sqg);
            tracer.position += self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
        }
        for (x, &k1, &k2, &k3, &k4) in self.state.point_vortices.iter_mut()
                                                 .zip(self.buffer.ks[0].iter())
                                                 .zip(self.buffer.ks[1].iter())
                                                 .zip(self.buffer.ks[2].iter())
                                                 .zip(self.buffer.ks[3].iter())
                                                 .map(|((((a, b), c), d), e)| (a, b, c, d, e)) {
            x.point_vortex.position += self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
        }
    }
}

pub struct Output {
    // nrow: usize,
    // ncol: usize,
    npv: usize,
    npt: usize,
    data: Vec<Data>
}

impl Output {
    pub fn write(&self) -> Result<Vec<u8>, crate::error::Error> {
        let nrow = self.npv + self.npt;
        let ncol = self.data.len() / nrow;
        let mut buf = vec![];
        let mut writer = npyz::WriteOptions::new()
            .default_dtype()
            .shape(&[nrow as u64, ncol as u64])
            .writer(&mut buf)
            .begin_nd()?;
        for i in 0..self.npv {
            writer.extend(self.data.iter().filter_map(|data| {
                match data {
                    Data::PointVortex(pv) if pv.index == i => Some(pv.point_vortex.position),
                    _ => None
                }
            }))?;
        }
        for i in 0..self.npt {
            writer.extend(self.data.iter().filter_map(|data| {
                match data {
                    Data::PassiveTracer(pt) if pt.index == i => Some(pt.position),
                    _ => None
                }
            }))?;
        }
        writer.finish()?;
        Ok(buf)
    }
}

pub struct Solver {
    nthread: usize,
    write_interval: usize,
    niter: usize,
    pub threads: Vec<TimeStepper>,
    chunk_size: usize,
    npv: usize,
    npt: usize,
}

impl Solver {
    pub fn new(problem: &Problem, threads: Option<u8>) -> Self {
        let niter = (problem.duration as f64 / problem.time_step).round() as usize;
        let write_interval = problem.write_interval.unwrap_or(1);
        let nthread = threads.and_then(NonZeroU8::new).map(NonZeroU8::get).unwrap_or(1) as usize;
        ThreadPoolBuilder::new().num_threads(nthread).build_global().unwrap();
        let npt = problem.passive_tracers.len();
        let npv = problem.point_vortices.len();
        let passive_tracers: Vec<_> = problem.passive_tracers.iter()
            .enumerate()
            .map(|(index, &position)| IndexedPassiveTracer { index, position })
            .collect();
        let point_vortices: Vec<_> = problem.point_vortices.iter()
            .enumerate()
            .map(|(index, &point_vortex)| IndexedPointVortex { index, point_vortex })
            .collect();
        let threads = passive_tracers.chunks((npt / nthread).max(1)).map(|pt| {
            let pv = &point_vortices;
            let rossby = problem.rossby;
            let dt = problem.time_step;
            let sqg = problem.sqg;
            TimeStepper::new(pv, pt, rossby, dt, sqg)
        }).collect();
        let chunk_size = 10_485_760;
        Solver { nthread, threads, write_interval, niter, npv, npt, chunk_size }
    }

    pub fn create_buffer(&self) -> Output {
        let n = (self.niter / self.write_interval) * (self.npt + self.npv);
        let data = repeat(Data::default()).take(n).collect();
        Output { data, npt: self.npt, npv: self.npv }
    }

    pub fn solve(&mut self, output: &mut Output) {
        output.data.par_chunks_mut(self.chunk_size).for_each(|chunk| {
            let n = chunk.len();
            let niter = n / self.threads.len();
            for _ in 0..niter {
                chunk.par_iter_mut().zip(self.threads.par_iter_mut())
                          .for_each(|(cell, thread)| {
                    loop {
                        match thread.output_queue.pop() {
                            Some(ObjectIndex::PointVortex(i)) => {
                                let pv = thread.state().point_vortices[i as usize];
                                *cell = Data::PointVortex(pv);
                                break
                            },
                            Some(ObjectIndex::PassiveTracer(i)) => {
                                let pt = thread.state().passive_tracers[i as usize];
                                *cell = Data::PassiveTracer(pt);
                                break
                            }
                            None => {
                                for _ in 0..self.write_interval {
                                    thread.next();
                                }
                            }
                        }
                    }
                })
            }
        })
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

fn ui(vtx: Vector, other_vtxs: &[IndexedPointVortex],
      rossby: f64, sqg: bool) -> Vector {
    let u0: Vector = other_vtxs
        .iter()
        .map(|&pv| pv.point_vortex.strength * u0ij(vtx, pv.point_vortex.position, sqg))
        .sum();
    let u1s: Vector = other_vtxs
        .iter()
        .map(|&pv| pv.point_vortex.strength.powi(2) * u1sij(vtx, pv.point_vortex.position, sqg))
        .sum();
    let u1p: Vector = other_vtxs
        .iter()
        .combinations(2)
        .map(|pvs| {
            let gamma1 = pvs[0].point_vortex.strength;
            let gamma2 = pvs[1].point_vortex.strength;
            let vtx1 = pvs[0].point_vortex.position;
            let vtx2 = pvs[1].point_vortex.position;
            gamma1 * gamma2 * u1pijk(vtx, vtx1, vtx2, sqg)
        })
        .sum();
    u0 + rossby * (u1s + u1p)
}
