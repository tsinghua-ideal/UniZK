use crate::config::enable_config::ENABLE_CONFIG;
use crate::starky::proof::StarkOpeningSet;
use crate::system::system::System;

use crate::kernel::hash_no_pad::{HashNoPad, HashNoPadConfig};
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::tree::Tree;
use crate::util::{FRI_PROOF_OF_WORK_ROUND, NUM_HASH_OUT_ELTS, SIZE_F, SPONGE_RATE as RATE};

use crate::plonk::proofs::OpeningSet;

pub struct Challenger {
    pub input_addrs: Vec<usize>,
    pub output_addr: usize,
    pub output_buffer_length: usize,
}

impl Challenger {
    pub fn new(output_addr: usize) -> Self {
        Self {
            input_addrs: Vec::new(),
            output_addr,
            output_buffer_length: 0,
        }
    }

    pub fn pow(&mut self) -> HashNoPad {
        let mut hk = HashNoPad::new(HashNoPadConfig {
            addr_input: self.input_addrs.clone(),
            addr_output: self.output_addr,
            input_length: vec![1; RATE],
            output_length: RATE,
        });
        unsafe {
            if ENABLE_CONFIG.hash {
                hk.drain.interval = 1.0;
                hk.write_request.num_lines = vec![FRI_PROOF_OF_WORK_ROUND];
            }
        }
        hk
    }

    pub fn observe_element(&mut self, sys: &mut System, addr: usize) {
        self.input_addrs.push(addr);
        if self.input_addrs.len() == RATE {
            let kernel = HashNoPad::new(HashNoPadConfig {
                addr_input: self.input_addrs.clone(),
                addr_output: self.output_addr,
                input_length: vec![1; RATE],
                output_length: RATE,
            });
            sys.run_once(&kernel);
            self.input_addrs.clear();
            self.output_buffer_length = RATE;
        }
    }

    pub fn observe_elements(&mut self, sys: &mut System, addr: usize, num_elements: usize) {
        for i in 0..num_elements {
            self.observe_element(sys, addr + i * SIZE_F);
        }
    }

    pub fn observe_hash(&mut self, sys: &mut System, addr_hash: usize) {
        self.observe_elements(sys, addr_hash, NUM_HASH_OUT_ELTS);
    }

    pub fn observe_cap(&mut self, sys: &mut System, addr_cap: usize, cap_height: usize) {
        for i in 0..Tree::num_caps(cap_height) {
            self.observe_hash(sys, addr_cap + i * Tree::DIGEST_LENGTH * SIZE_F)
        }
    }

    pub fn get_challenge(&mut self, sys: &mut System, addr: usize) {
        if self.output_buffer_length == 0 {
            if self.input_addrs.is_empty() {
                self.input_addrs.extend(vec![0; RATE]);
            }
            let kernel = HashNoPad::new(HashNoPadConfig {
                addr_input: self.input_addrs.clone(),
                addr_output: self.output_addr,
                input_length: vec![1; RATE],
                output_length: RATE,
            });
            sys.run_once(&kernel);
            self.input_addrs.clear();
            self.output_buffer_length = RATE;
        }

        let mut kernel = MemCpy::new(
            MemCpyConfig {
                addr_input: self.output_addr + (self.output_buffer_length - 1) * SIZE_F,
                addr_output: addr,
                input_length: 1,
            },
            unsafe { ENABLE_CONFIG.hash },
        );
        kernel.prefetch.addr.clear();
        self.output_buffer_length -= 1;
        sys.run_once(&kernel);
    }

    pub fn get_n_challenges(&mut self, sys: &mut System, addr: usize, n: usize) {
        if self.output_buffer_length >= n {
            let mut kernel = MemCpy::new(
                MemCpyConfig {
                    addr_input: self.output_addr + (self.output_buffer_length - n) * SIZE_F,
                    addr_output: addr,
                    input_length: n,
                },
                unsafe { ENABLE_CONFIG.hash },
            );
            kernel.prefetch.clear();
            kernel.read_request.clear();
            self.output_buffer_length -= n;
            sys.run_once(&kernel);
        } else {
            for i in 0..n {
                self.get_challenge(sys, self.output_addr + i * SIZE_F);
            }
        }
    }

    pub fn observe_extension_element<const D: usize>(&mut self, sys: &mut System, addr: usize) {
        for i in 0..D {
            self.observe_element(sys, addr + i * SIZE_F);
        }
    }

    pub fn observe_extension_elements<const D: usize>(
        &mut self,
        sys: &mut System,
        addr: usize,
        num_elements: usize,
    ) {
        for i in 0..num_elements {
            self.observe_extension_element::<D>(sys, addr + i * D * SIZE_F);
        }
    }

    pub fn observe_openings<const D: usize>(&mut self, sys: &mut System, openings: &OpeningSet<D>) {
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_constants,
            openings.constants_length,
        );
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_plonk_sigmas,
            openings.plonk_sigmas_length,
        );

        self.observe_extension_elements::<D>(sys, openings.addr_wires, openings.wires_length);

        self.observe_extension_elements::<D>(sys, openings.addr_plonk_zs, openings.plonk_zs_length);
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_partial_products,
            openings.partial_products_length,
        );
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_quotient_polys,
            openings.quotient_polys_length,
        );
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_plonk_zs_next,
            openings.plonk_zs_next_length,
        );
    }
    pub fn observe_stark_openings<const D: usize>(
        &mut self,
        sys: &mut System,
        openings: &StarkOpeningSet<D>,
    ) {
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_local_values,
            openings.local_values_length,
        );
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_quotient_polys,
            openings.quotient_polys_length,
        );
        self.observe_extension_elements::<D>(
            sys,
            openings.addr_next_values,
            openings.next_values_length,
        );
    }

    pub fn get_extension_challenge<const D: usize>(&mut self, sys: &mut System, addr: usize) {
        self.get_n_challenges(sys, addr, D)
    }
}
