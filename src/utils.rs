use chrono::{DateTime, Local};
use main_error::MainError;
use npyz::{npz::NpzWriter, WriterBuilder};

use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufWriter;
use std::path::Path;


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

pub struct Writer {
    file_handle: NpzWriter<BufWriter<File>>
}

impl Writer {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, MainError> {
        let file_handle = npyz::npz::NpzWriter::create(path)?;
        Ok(Writer { file_handle })
    }

    pub fn writez<T, U>(&mut self, label: &str, dim: &[u64], data: T)
        -> Result<(), MainError>
        where T: IntoIterator<Item=U>,
              U: npyz::AutoSerialize + npyz::Serialize {
        let mut writer = self.file_handle.array(label, Default::default())?
            .default_dtype()
            .shape(dim)
            .begin_nd()?;
        writer.extend(data.into_iter())?;
        writer.finish()?;
        Ok(())
    }
}

