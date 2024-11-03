use log::debug;
use plonky2::util::log2_strict;

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::kernel::Kernel;
use crate::util::{bit_reverse, SIZE_F};
#[derive(Debug, Clone, Copy)]
pub struct TransposeConfig {
    pub addr_input: usize,
    pub addr_output: usize,
    pub width: usize,
    pub height: usize,
    pub start: usize,
    pub end: usize,
    pub reverse: bool,
    pub extension: usize,
}

// todo eliminate real fetch
pub struct Transpose {
    pub config: TransposeConfig,
    pub prefetch: Fetch,
    pub drain: Fetch,
    pub write_request: Request,
    pub read_request: Request,
}

impl Kernel for Transpose {
    fn create_drain(&mut self) {
        let array_length = unsafe { 1 << (ARCH_CONFIG.array_length >> 1) };
        let num_blocks_on_chip = unsafe {
            ARCH_CONFIG.num_elems() / array_length / array_length / self.config.extension
        };
        let mut addrs = Vec::new();
        let mut block_cnt = 0;
        let height = self.config.width;
        let width = self.config.height; // transpose
        for width_chunk in (0..width).step_by(array_length) {
            for height_chunk in (self.config.start..self.config.end).step_by(array_length) {
                for i in height_chunk..(height_chunk + array_length).min(height) {
                    let block_length = array_length.min(width - width_chunk);
                    let i_res = if self.config.reverse {
                        bit_reverse(i, log2_strict(height))
                    } else {
                        i
                    };
                    addrs.push(
                        (width_chunk..width_chunk + block_length)
                            .map(|j| {
                                self.config.addr_output
                                    + (i_res * width + j) * SIZE_F * self.config.extension
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                block_cnt += 1;
                if block_cnt == num_blocks_on_chip {
                    self.drain.push(addrs.clone());
                    self.write_request.push(addrs.clone());
                    addrs.clear();
                    block_cnt = 0;
                }
            }
        }
        if !addrs.is_empty() {
            self.write_request.push(addrs.clone());
            self.drain.push(addrs);
        }
        self.drain.delay = vec![0; self.drain.len()];
        self.drain.mergable = true;
        self.drain.interval = 1.0;
        self.drain.systolic = true;
    }
    fn create_read_request(&mut self) {}
    fn create_write_request(&mut self) {}
    fn create_prefetch(&mut self) {
        let array_length = unsafe { 1 << (ARCH_CONFIG.array_length >> 1) };
        let num_blocks_on_chip = unsafe {
            ARCH_CONFIG.num_elems() / array_length / array_length / self.config.extension
        };
        let mut addrs = Vec::new();
        let mut block_cnt = 0;
        for height_chunk in (0..self.config.height).step_by(array_length) {
            for width_chunk in (self.config.start..self.config.end).step_by(array_length) {
                for i in height_chunk..(height_chunk + array_length).min(self.config.height) {
                    let block_length = array_length.min(self.config.width - width_chunk);
                    addrs.push(
                        (width_chunk..width_chunk + block_length)
                            .map(|j| {
                                self.config.addr_input
                                    + (i * self.config.width + j) * SIZE_F * self.config.extension
                            })
                            .collect::<Vec<_>>(),
                    );
                }
                block_cnt += 1;
                if block_cnt == num_blocks_on_chip {
                    self.prefetch.push(addrs.clone());
                    self.read_request.push(addrs.clone());
                    addrs.clear();
                    block_cnt = 0;
                }
            }
        }
        if !addrs.is_empty() {
            self.read_request.push(addrs.clone());
            self.prefetch.push(addrs);
        }
        self.prefetch.mergable = true;
        self.prefetch.delay = vec![0; self.prefetch.len()];
        self.prefetch.interval = 0.0;
        self.prefetch.systolic = true;
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
        debug!("Kernel config: {:?}", self.config);
    }
    fn get_kernel_type(&self) -> String {
        String::from("Transpose")
    }

    fn get_computation(&self) -> usize {
        0
    }
}

impl Transpose {
    pub fn new(config: TransposeConfig) -> Self {
        let input_left = config.addr_input;
        let input_right =
            config.addr_input + config.width * config.height * SIZE_F * config.extension;
        let output_left = config.addr_output;
        let output_right = config.addr_output
            + (config.end - config.start) * config.height * SIZE_F * config.extension;
        assert!(
            input_right <= output_left || output_right <= input_left,
            "Transpose config {:?} has overlap",
            config
        );

        assert!(config.extension != 0);
        let mut k = Self {
            config,
            prefetch: Fetch::new(FetchType::Read),
            drain: Fetch::new(FetchType::Write),
            write_request: Request::new(),
            read_request: Request::new(),
        };
        if unsafe { ENABLE_CONFIG.other } {
            k.init();
        }
        k
    }
}
