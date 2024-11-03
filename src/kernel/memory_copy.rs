use log::debug;

use crate::trace::trace::{Fetch, FetchType, Request};
use crate::config::arch_config::ARCH_CONFIG;
use crate::kernel::kernel::Kernel;
use crate::util::SIZE_F;
#[derive(Debug, Clone)]
pub struct MemCpyConfig {
    pub addr_input: usize,
    pub addr_output: usize,
    pub input_length: usize, // number of elements
}

// todo eliminate real fetch
pub struct MemCpy {
    pub config: MemCpyConfig,
    pub prefetch: Fetch,
    pub drain: Fetch,
    pub write_request: Request,
    pub read_request: Request,
}

impl Kernel for MemCpy {
    fn create_drain(&mut self) {
        let addrs = (0..self.config.input_length)
            .map(|i| self.config.addr_output + i * SIZE_F)
            .collect::<Vec<_>>();
        unsafe {
            let num_elems = ARCH_CONFIG.num_elems();
            addrs.chunks(num_elems).for_each(|chunk| {
                self.drain.push(
                    chunk
                        .chunks(ARCH_CONFIG.array_length)
                        .map(|c| c.to_vec())
                        .collect(),
                );
                self.write_request.push(
                    chunk
                        .chunks(ARCH_CONFIG.array_length)
                        .map(|c| c.to_vec())
                        .collect(),
                );
            });
        }

        self.drain.mergable = true;
        self.drain.interval = 0.0;
        self.drain.delay = vec![0; self.drain.len()];
    }
    fn create_read_request(&mut self) {}
    fn create_write_request(&mut self) {}
    fn create_prefetch(&mut self) {
        let addrs = (0..self.config.input_length)
            .map(|i| self.config.addr_input + i * SIZE_F)
            .collect::<Vec<_>>();
        unsafe {
            let num_elems = ARCH_CONFIG.num_elems();
            addrs.chunks(num_elems).for_each(|chunk| {
                self.prefetch.push(
                    chunk
                        .chunks(ARCH_CONFIG.array_length)
                        .map(|c| c.to_vec())
                        .collect(),
                );
                self.read_request.push(
                    chunk
                        .chunks(ARCH_CONFIG.array_length)
                        .map(|c| c.to_vec())
                        .collect(),
                );
            });
        }
        self.prefetch.mergable = true;
        self.prefetch.interval = 0.0;
        self.prefetch.delay = vec![0; self.prefetch.len()];
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
        String::from("Memcpy")
    }
    fn get_computation(&self) -> usize {
        0
    }
}

impl MemCpy {
    pub fn new(config: MemCpyConfig, enable: bool) -> Self {
        let mut k = Self {
            config,
            prefetch: Fetch::new(FetchType::Read),
            drain: Fetch::new(FetchType::Write),
            write_request: Request::new(),
            read_request: Request::new(),
        };
        if enable {
            k.init();
        }
        k
    }

    /// translate the address from output to input
    pub fn addr_trans(&self, addr: &(u64, u64)) -> Vec<(u64, u64)> {
        let mut res = Vec::new();
        let (addr1, addr2) = addr.clone();
        let addr_start = self.config.addr_output as u64;
        let addr_end = (self.config.addr_output + self.config.input_length * SIZE_F) as u64;

        let overlap_start = addr1.max(addr_start);
        let overlap_end = addr2.min(addr_end);

        let offset =
            |addr: u64| addr - self.config.addr_output as u64 + self.config.addr_input as u64;
        if overlap_start < overlap_end {
            res.push((offset(overlap_start), offset(overlap_end)));
            if addr1 < overlap_start {
                res.push((addr1, overlap_start - 1));
            }
            if addr2 > overlap_end {
                res.push((overlap_end + 1, addr2));
            }
        } else {
            res.push(addr.clone());
        }
        res
    }
}
