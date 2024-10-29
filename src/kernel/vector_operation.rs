use crate::memory::memory_allocator::MemAlloc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecOpType {
    ADD,
    SUB,
    MUL,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VecOpSrc {
    VV,
    VS,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VecOpConfig {
    pub vector_length: usize,  // length of the vector
    pub addr_input_0: usize,   // base address of the input 0
    pub addr_input_1: usize,   // base address of the input 1
    pub addr_output: usize,    // base address of the output
    pub is_final_output: bool, // is the output the final output
    pub op_type: VecOpType,    // type of the operation
    pub op_src: VecOpSrc,      // source of the operation
}

use crate::util::SIZE_F;
impl VecOpConfig {
    pub fn is_mul(&self) -> bool {
        match self.op_type {
            VecOpType::MUL => true,
            _ => false,
        }
    }

    pub fn is_add(&self) -> bool {
        match self.op_type {
            VecOpType::ADD => true,
            _ => false,
        }
    }

    pub fn is_sub(&self) -> bool {
        match self.op_type {
            VecOpType::SUB => true,
            _ => false,
        }
    }

    pub fn is_add_or_sub(&self) -> bool {
        self.is_add() || self.is_sub()
    }

    pub fn inv(
        mem: &mut MemAlloc,
        vector_length: usize, // length of the vector
        addr_input: usize,
        addr_output: usize,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();

        mem.alloc("t2_3_12_30_63", vector_length * SIZE_F);
        let addr_t2_3_12_30_63 = mem.get_addr("t2_3_12_30_63").unwrap();

        mem.alloc("t6_31", vector_length * SIZE_F);
        let addr_t6_31 = mem.get_addr("t6_31").unwrap();

        mem.alloc("t24", vector_length * SIZE_F);
        let addr_t24 = mem.get_addr("t24").unwrap();

        // t2
        vec_ops.extend(Self::exp_acc::<1>(
            vector_length,
            addr_input,
            addr_input,
            addr_t2_3_12_30_63,
            false,
        ));

        // t3
        vec_ops.extend(Self::exp_acc::<1>(
            vector_length,
            addr_t2_3_12_30_63,
            addr_input,
            addr_t2_3_12_30_63,
            false,
        ));

        // t6
        vec_ops.extend(Self::exp_acc::<3>(
            vector_length,
            addr_t2_3_12_30_63,
            addr_t2_3_12_30_63,
            addr_t6_31,
            false,
        ));

        // t12
        vec_ops.extend(Self::exp_acc::<6>(
            vector_length,
            addr_t6_31,
            addr_t6_31,
            addr_t2_3_12_30_63,
            false,
        ));

        // t24
        vec_ops.extend(Self::exp_acc::<12>(
            vector_length,
            addr_t2_3_12_30_63,
            addr_t2_3_12_30_63,
            addr_t24,
            false,
        ));

        // t30
        vec_ops.extend(Self::exp_acc::<6>(
            vector_length,
            addr_t24,
            addr_t6_31,
            addr_t2_3_12_30_63,
            false,
        ));

        // t31
        vec_ops.extend(Self::exp_acc::<1>(
            vector_length,
            addr_t2_3_12_30_63,
            addr_input,
            addr_t6_31,
            false,
        ));

        // t63 base^111111111111111111111111111111101111111111111111111111111111111
        vec_ops.extend(Self::exp_acc::<32>(
            vector_length,
            addr_t6_31,
            addr_t6_31,
            addr_t2_3_12_30_63,
            false,
        ));

        //     base^1111111111111111111111111111111011111111111111111111111111111111
        vec_ops.extend(Self::exp_acc::<1>(
            vector_length,
            addr_t2_3_12_30_63,
            addr_input,
            addr_output,
            is_final_output,
        ));

        mem.free("t2_3_12_30_63");
        mem.free("t6_31");
        mem.free("t24");

        vec_ops
    }

    fn exp_acc<const N: usize>(
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();

        for i in 0..N {
            let ai = if i == 0 { addr_input_0 } else { addr_output };
            vec_ops.push(Self::square(vector_length, ai, addr_output, false));
        }
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_output,
            addr_input_1: addr_input_1,
            addr_output: addr_output,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: is_final_output,
        });

        vec_ops
    }

    fn square(
        vector_length: usize,
        addr_input: usize,
        addr_output: usize,
        is_final_output: bool,
    ) -> VecOpConfig {
        VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_input,
            addr_input_1: addr_input,
            addr_output: addr_output,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: is_final_output,
        }
    }

    pub fn delay(&self) -> usize {
        match self.op_type {
            VecOpType::ADD => 1,
            VecOpType::SUB => 1,
            VecOpType::MUL => 2,
        }
    }
}

pub struct VecOpExtension<const D: usize> {}
impl<const D: usize> VecOpExtension<D> {
    pub fn add(
        vector_length: usize,
        addr_input_0: usize, //  D * vector_length *SIZE_F
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        match D {
            2 => VecOpExtension2::add(
                vector_length,
                addr_input_0,
                addr_input_1,
                addr_output,
                op_src,
                is_final_output,
            ),
            _ => panic!("Unsupported D"),
        }
    }
    pub fn sub(
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        match D {
            2 => VecOpExtension2::sub(
                vector_length,
                addr_input_0,
                addr_input_1,
                addr_output,
                op_src,
                is_final_output,
            ),
            _ => panic!("Unsupported D"),
        }
    }
    pub fn mul(
        mem: &mut MemAlloc,
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        match D {
            2 => VecOpExtension2::mul(
                mem,
                vector_length,
                addr_input_0,
                addr_input_1,
                addr_output,
                op_src,
                is_final_output,
            ),
            _ => panic!("Unsupported D"),
        }
    }

    pub fn add_delay() -> usize {
        match D {
            2 => VecOpExtension2::ADD_DELAY,
            _ => panic!("Unsupported D"),
        }
    }
    pub fn sub_delay() -> usize {
        match D {
            2 => VecOpExtension2::SUB_DELAY,
            _ => panic!("Unsupported D"),
        }
    }
    pub fn mul_delay() -> usize {
        match D {
            2 => VecOpExtension2::MUL_DELAY,
            _ => panic!("Unsupported D"),
        }
    }

    pub fn scalar_mul(
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        for i in 0..D {
            vec_ops.push(VecOpConfig {
                vector_length: vector_length,
                addr_input_0: addr_input_0 + i * vector_length * SIZE_F,
                addr_input_1: addr_input_1,
                addr_output: addr_output,
                op_type: VecOpType::MUL,
                op_src: op_src,
                is_final_output: is_final_output,
            });
        }
        vec_ops
    }

    pub fn exp_u64(
        mem: &mut MemAlloc,
        _addr_input: usize,
        addr_res: usize,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let addr_current = mem.alloc("current", D * SIZE_F).unwrap();
        for _j in 0..64 {
            vec_ops.extend(Self::mul(
                mem,
                1,
                addr_res,
                addr_current,
                addr_res,
                VecOpSrc::VS,
                is_final_output,
            ));
            vec_ops.extend(Self::mul(
                mem,
                1,
                addr_current,
                addr_current,
                addr_current,
                VecOpSrc::VS,
                false,
            ));
        }
        mem.free("current");
        vec_ops
    }
}

pub struct VecOpExtension2 {}
impl VecOpExtension2 {
    const ADD_DELAY: usize = 2;
    const SUB_DELAY: usize = 2;
    const MUL_DELAY: usize = 5;

    fn add(
        vector_length: usize,
        addr_input_0: usize, //  2 * vector_length *SIZE_F
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let is_vs = op_src == VecOpSrc::VS;
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_input_0,
            addr_input_1: addr_input_1,
            addr_output: addr_output,
            op_type: VecOpType::ADD,
            op_src: op_src,
            is_final_output: is_final_output,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: if addr_input_0 == 0 {
                0
            } else {
                addr_input_0 + vector_length * SIZE_F
            },
            addr_input_1: if addr_input_1 == 0 {
                0
            } else if is_vs {
                addr_input_1 + SIZE_F
            } else {
                addr_input_1 + vector_length * SIZE_F
            },
            addr_output: addr_output,
            op_type: VecOpType::ADD,
            op_src: op_src,
            is_final_output: is_final_output,
        });
        vec_ops
    }
    fn sub(
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let is_vs = op_src == VecOpSrc::VS;
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_input_0,
            addr_input_1: addr_input_1,
            addr_output: addr_output,
            op_type: VecOpType::SUB,
            op_src: op_src,
            is_final_output: is_final_output,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: if addr_input_0 == 0 {
                0
            } else {
                addr_input_0 + vector_length * SIZE_F
            },
            addr_input_1: if addr_input_1 == 0 {
                0
            } else if is_vs {
                addr_input_1 + SIZE_F
            } else {
                addr_input_1 + vector_length * SIZE_F
            },
            addr_output: addr_output,
            op_type: VecOpType::SUB,
            op_src: op_src,
            is_final_output: is_final_output,
        });
        vec_ops
    }
    fn mul(
        mem: &mut MemAlloc,
        vector_length: usize,
        addr_input_0: usize,
        addr_input_1: usize,
        addr_output: usize,
        op_src: VecOpSrc,
        is_final_output: bool,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let is_vs = op_src == VecOpSrc::VS;
        let a0 = addr_input_0;
        let a1 = if addr_input_0 == 0 {
            0
        } else {
            addr_input_0 + vector_length * SIZE_F
        };
        let b0 = addr_input_1;
        let b1 = if addr_input_1 == 0 {
            0
        } else if is_vs {
            addr_input_1 + SIZE_F
        } else {
            addr_input_1 + vector_length * SIZE_F
        };
        // a0 * b0
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: a0,
            addr_input_1: b0,
            addr_output: addr_output,
            op_type: VecOpType::MUL,
            op_src: op_src,
            is_final_output: is_final_output,
        });
        let addr_exmul_0 = mem.alloc("exmul_0", vector_length * SIZE_F).unwrap();
        // W * a1
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: a1,
            addr_input_1: 0,
            addr_output: addr_exmul_0,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        // _ * b1
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_exmul_0,
            addr_input_1: b1,
            addr_output: addr_exmul_0,
            op_type: VecOpType::MUL,
            op_src: op_src,
            is_final_output: false,
        });
        // a0 * b0 + W * a1 * b1
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_output,
            addr_input_1: addr_exmul_0,
            addr_output: addr_output,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: is_final_output,
        });

        // a0 * b1
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: a0,
            addr_input_1: b1,
            addr_output: addr_exmul_0,
            op_type: VecOpType::MUL,
            op_src: op_src,
            is_final_output: false,
        });
        // a1 * b0
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: a1,
            addr_input_1: b0,
            addr_output: addr_output + vector_length * SIZE_F,
            op_type: VecOpType::MUL,
            op_src: op_src,
            is_final_output: false,
        });
        // a0 * b1 + a1 * b0
        vec_ops.push(VecOpConfig {
            vector_length: vector_length,
            addr_input_0: addr_exmul_0,
            addr_input_1: addr_output + vector_length * SIZE_F,
            addr_output: addr_output + vector_length * SIZE_F,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: is_final_output,
        });

        mem.free("exmul_0");

        vec_ops
    }
}
