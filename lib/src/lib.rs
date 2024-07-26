use chrono::{DateTime, Local};

use std::io;
use std::io::prelude::*;

pub mod config;
pub mod error;
pub mod problem;

#[derive(Clone, Copy)]
pub struct Progress {
    pub threshold: f64,
    pub niter: usize,
    pub step: usize,
    pub start_time: DateTime<Local>
}

impl Progress {
    pub fn new(niter: usize) -> Self {
        print!("0.0% complete\r");
        io::stdout().flush().unwrap();
        let start_time = Local::now();
        Progress { threshold: 0., step: 1, niter, start_time }
    }

    pub fn step(&mut self, stdout: bool) {
        if stdout {
            let niter_f64 = self.niter as f64;
            let step_f64 = self.step as f64;
            let percent_done = (step_f64 + 1.) * 100. / niter_f64;
            let hundredths = (percent_done * 10.).floor();
            if hundredths > self.threshold {
                self.threshold = hundredths;
                let now = Local::now();
                let elapsed_time = now - self.start_time;
                let unit_time = elapsed_time / (percent_done * 100.).round() as i32;
                let time_left = unit_time * ((100. - percent_done) * 100.).round() as i32;
                let eta = now + time_left;
                print!("{:.1}% complete. Estimated completion time: {}\r",
                       percent_done, eta.format("%m-%d-%Y %H:%M:%S"));
                io::stdout().flush().unwrap();
            }
        }
        self.step += 1;
    }
}
