use log::info;

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::kernel::Kernel;
use crate::util::{HASH_COMPUTATION, SINGLE_HASH_DELAY, SIZE_F, SPONGE_RATE};
#[derive(Debug, Clone)]
pub struct HashNoPadConfig {
    pub addr_input: Vec<usize>,
    pub addr_output: usize,
    pub input_length: Vec<usize>,
    pub output_length: usize,
}

#[derive(Debug, Clone)]
pub struct HashNoPad {
    pub config: HashNoPadConfig,
    pub prefetch: Fetch,
    pub drain: Fetch,
    pub write_request: Request,
    pub read_request: Request,
}

impl Kernel for HashNoPad {
    fn create_prefetch(&mut self) {
        let num_inputs = self.config.input_length.iter().sum();
        let mut addr = Vec::with_capacity(num_inputs);
        for (a, i) in self
            .config
            .addr_input
            .iter()
            .zip(self.config.input_length.iter())
        {
            for j in 0..*i {
                addr.push(a + j * SIZE_F);
            }
        }
        self.prefetch.push(
            addr.chunks(SPONGE_RATE)
                .map(|chunk| chunk.to_vec())
                .collect(),
        );
        self.read_request.push(
            addr.chunks(SPONGE_RATE)
                .map(|chunk| chunk.to_vec())
                .collect(),
        );
        self.prefetch.delay = vec![0; self.prefetch.len()];
        self.prefetch.interval = 0.0;
        self.prefetch.mergable = true;
        self.prefetch.systolic = true;
    }
    fn create_read_request(&mut self) {}
    fn create_write_request(&mut self) {}
    fn create_drain(&mut self) {
        self.drain.push(vec![(0..self.config.output_length)
            .map(|i| self.config.addr_output + i * SIZE_F)
            .collect()]);
        self.write_request.push(vec![(0..self.config.output_length)
            .map(|i| self.config.addr_output + i * SIZE_F)
            .collect()]);
        self.drain.delay = vec![SINGLE_HASH_DELAY; self.drain.len()];
        self.drain.interval = SINGLE_HASH_DELAY as f32;

        self.drain.mergable = true;
        self.drain.systolic = true;
    }

    fn get_prefetch(&self) -> Fetch {
        self.prefetch.clone()
    }
    fn get_drain(&self) -> Fetch {
        self.drain.clone()
    }
    fn get_read_request(&self) -> Request {
        self.read_request.clone()
    }
    fn get_write_request(&self) -> Request {
        self.write_request.clone()
    }
    fn log(&self) {
        info!("Kernel config: {:?}", self.config);
    }
    fn get_kernel_type(&self) -> String {
        String::from("Hash")
    }
    fn get_computation(&self) -> usize {
        self.read_request.num_request_lines() * HASH_COMPUTATION
    }
}

impl HashNoPad {
    pub fn new(config: HashNoPadConfig) -> Self {
        let mut hash_no_pad = Self {
            config,
            prefetch: Fetch::new(FetchType::Read),
            drain: Fetch::new(FetchType::Write),
            write_request: Request::new(),
            read_request: Request::new(),
        };
        if unsafe { ENABLE_CONFIG.hash } {
            hash_no_pad.init();
        }
        hash_no_pad
    }
}
