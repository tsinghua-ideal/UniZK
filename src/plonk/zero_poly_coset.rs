use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};

use crate::memory::memory_allocator::MemAlloc;
use crate::system::system::System;
use crate::util::SIZE_F;

pub struct ZeroPolyOnCoset {
    pub rate: usize,
    addr_evals: usize,
    pub addr_inverses: usize,
}

impl ZeroPolyOnCoset {
    pub fn new(sys: &mut System, rate_bits: usize) -> Self {
        let rate = 1 << rate_bits;
        let addr_g_pow_n = sys.mem.alloc("g_pow_n_cpu", SIZE_F).unwrap();
        let addr_evals = sys.mem.alloc("addr_evals_cpu", rate * SIZE_F).unwrap(); // subgroup from cpu
        let addr_inverses = sys.mem.alloc("addr_inverses", rate * SIZE_F).unwrap();

        let mut vec_ops = Vec::new();
        // g_pow_n * x
        vec_ops.push(VecOpConfig {
            vector_length: rate,
            addr_input_0: addr_evals,
            addr_input_1: addr_g_pow_n,
            addr_output: addr_evals,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        // _ - F::ONE
        vec_ops.push(VecOpConfig {
            vector_length: rate,
            addr_input_0: addr_evals,
            addr_input_1: 0, // const scalar
            addr_output: addr_evals,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
            is_final_output: true,
        });
        vec_ops.extend(VecOpConfig::inv(
            &mut sys.mem,
            rate,
            addr_evals,
            addr_inverses,
            true,
        ));
        let kernel = VectorChain::new(vec_ops, &sys.mem);
        sys.run_once(&kernel);

        Self {
            rate,
            addr_evals,
            addr_inverses,
        }
    }

    pub fn eval_l_0(&self, sys: &mut System, n: usize, addr_x: usize, addr_res: usize) {
        let mut vec_ops = Vec::new();

        // x - F::ONE
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_x,
            addr_input_1: 0, // const scalar
            addr_output: addr_res,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        // self.n * _
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_res,
            addr_input_1: 0,
            addr_output: addr_res,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        for i in (0..n).step_by(self.rate) {
            let vl = (n - i).min(self.rate);
            vec_ops.push(VecOpConfig {
                vector_length: vl,
                addr_input_0: self.addr_evals,
                addr_input_1: addr_res,
                addr_output: addr_res,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }
        vec_ops.extend(VecOpConfig::inv(&mut sys.mem, n, addr_res, addr_res, true));

        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
    }

    pub fn eval_inverse(&self, i: usize) -> usize {
        self.addr_inverses + (i % self.rate) * SIZE_F
    }

    pub fn preload(&self, mem: &mut MemAlloc) {
        mem.preload(self.addr_evals, self.rate);
        mem.preload(self.addr_inverses, self.rate);
    }
}
