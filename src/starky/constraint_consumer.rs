use crate::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};
use crate::util::{BATCH_SIZE, SIZE_F};

pub struct ConstraintConsumer {
    alphas_len: usize,
    addr_alphas: usize,
    addr_constraint_accs: Vec<usize>,
    addr_z_last: usize,
    addr_lagrange_basis_first: usize,
    addr_lagrange_basis_last: usize,
}

impl ConstraintConsumer {
    pub fn new(
        alphas_len: usize,
        addr_alphas: usize,
        addr_z_last: usize,
        addr_lagrange_basis_first: usize,
        addr_lagrange_basis_last: usize,
        addr_constraint_accs: Vec<usize>,
    ) -> Self {
        Self {
            alphas_len,
            addr_alphas,
            addr_constraint_accs,
            addr_z_last,
            addr_lagrange_basis_first,
            addr_lagrange_basis_last,
        }
    }

    pub fn constraint_transition(&self, addr_constraint: usize) -> Vec<VecOpConfig> {
        let mut res = Vec::new();
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: self.addr_z_last,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(self.constraint(addr_constraint));
        res
    }

    pub fn constraint(&self, addr_constraint: usize) -> Vec<VecOpConfig> {
        let mut res = Vec::new();
        for i in 0..self.alphas_len {
            let addr_alpha = self.addr_alphas + i * SIZE_F;
            let addr_constraint_acc = self.addr_constraint_accs[i];
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: if i != 0 { addr_constraint_acc } else { 0 },
                addr_input_1: addr_alpha,
                addr_output: addr_constraint_acc,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_constraint_acc,
                addr_output: addr_constraint,
                is_final_output: i == self.alphas_len - 1,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            });
        }
        res
    }

    pub fn constraint_first_row(&self, addr_constraint: usize) -> Vec<VecOpConfig> {
        let mut res = Vec::new();
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: self.addr_lagrange_basis_first,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(self.constraint(addr_constraint));
        res
    }

    pub fn constraint_last_row(&self, addr_constraint: usize) -> Vec<VecOpConfig> {
        let mut res = Vec::new();
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: self.addr_lagrange_basis_last,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(self.constraint(addr_constraint));
        res
    }

    pub fn accumulators(&self) -> &Vec<usize> {
        &self.addr_constraint_accs
    }
}
