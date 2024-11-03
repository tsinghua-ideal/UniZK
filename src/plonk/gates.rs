use log::debug;
use plonky2::gates::gate::GateRef;

use crate::kernel::vector_operation::{VecOpConfig, VecOpExtension, VecOpSrc, VecOpType};
use crate::memory::memory_allocator::MemAlloc;
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::system::system::System;
use crate::util::{ceil_div_usize, B, SIZE_F, SPONGE_WIDTH};
use core::ops::Range;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use std::time::Instant;
pub fn eval_filtered_base_batch<F: RichField + Extendable<D>, const D: usize>(
    sys: &mut System,
    gate_ref: &GateRef<F, D>,
    mut vars_batch: EvaluationVarsBaseBatch,
    row: usize,
    selector_index: usize,
    addr_group_range: usize,
    group_range: Range<usize>,
    num_selectors: usize,
    num_lookup_selectors: usize,
    addr_res_batch: usize,
) -> Vec<VecOpConfig> {
    let mut _start = Instant::now();
    let addr_sub = sys.mem.alloc("sub", vars_batch.len() * SIZE_F).unwrap();
    let addr_filters = sys.mem.alloc("filters", vars_batch.len() * SIZE_F).unwrap();
    let mut vec_ops = Vec::new();

    let gate = &gate_ref.0;

    for i in 0..group_range.len() {
        if i != row {
            vec_ops.push(VecOpConfig {
                vector_length: vars_batch.len(),
                addr_input_0: vars_batch.addr_local_constants
                    + selector_index * vars_batch.len() * SIZE_F,
                addr_input_1: addr_group_range + i * SIZE_F,
                addr_output: addr_sub,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_batch.len(),
                addr_input_0: addr_sub,
                addr_input_1: addr_filters,
                addr_output: addr_filters,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: i == group_range.len() - 1,
            });
        }
    }
    vars_batch.addr_local_constants +=
        (num_selectors + num_lookup_selectors) * vars_batch.len() * SIZE_F;

    debug!("Gate: {:?}", gate.id());
    if let Some(index) = gate.id().find("Gate") {
        let substring = &gate.id()[..index + 4];
        match substring {
            "ArithmeticGate" => vec_ops.extend(eval_unfiltered_base_packed_arithmetic(
                &mut sys.mem,
                vars_batch,
                addr_res_batch,
                get_params(gate.id(), "num_ops"),
            )),
            "BaseSumGate" => vec_ops.extend(eval_unfiltered_base_packed_base_sum::<B>(
                &mut sys.mem,
                vars_batch,
                addr_res_batch,
                get_params(gate.id(), "num_limbs"),
            )),
            "ConstantGate" => vec_ops.extend(eval_unfiltered_base_packed_constant(
                vars_batch,
                addr_res_batch,
                get_params(gate.id(), "num_consts"),
            )),
            "ExponentiationGate" => vec_ops.extend(eval_unfiltered_base_packed_exponentiation(
                &mut sys.mem,
                vars_batch,
                addr_res_batch,
                get_params(gate.id(), "num_power_bits"),
            )),
            "PublicInputGate" => vec_ops.extend(eval_unfiltered_base_packed_public_input(
                vars_batch,
                addr_res_batch,
            )),
            "RandomAccessGate" => vec_ops.extend(eval_unfiltered_base_packed_random_access(
                &mut sys.mem,
                vars_batch,
                addr_res_batch,
                get_params(gate.id(), "bits"),
                get_params(gate.id(), "num_copies"),
                get_params(gate.id(), "num_extra_constants"),
            )),
            "PoseidonGate" => vec_ops.extend(eval_unfiltered_base_poseidon(
                &mut sys.mem,
                vars_batch,
                addr_res_batch,
            )),
            "NoopGate" => {
                // Do nothing
            }
            "U32AddManyGate" => {
                let num_addends = get_params(gate.id(), "num_addends");
                let num_ops = get_params(gate.id(), "num_ops");
                vec_ops.extend(eval_unfiltered_base_packed_u32_add_many(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_addends,
                    num_ops,
                ));
            }
            "U32RangeCheckGate" => {
                let num_input_limbs = get_params(gate.id(), "num_input_limbs");
                vec_ops.extend(eval_unfiltered_base_packed_u32_range_check(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_input_limbs,
                ));
            }
            "ComparisonGate" => {
                let num_bits = get_params(gate.id(), "num_bits");
                let num_chunks = get_params(gate.id(), "num_chunks");
                vec_ops.extend(eval_unfiltered_base_packed_comparison(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_bits,
                    num_chunks,
                ));
            }
            "U32ArithmeticGate" => {
                let num_ops = get_params(gate.id(), "num_ops");
                vec_ops.extend(eval_unfiltered_base_packed_u32_arithmetic(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_ops,
                ));
            }
            "U32SubtractionGate" => {
                let num_ops = get_params(gate.id(), "num_ops");
                vec_ops.extend(eval_unfiltered_base_packed_u32_subtraction(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_ops,
                ));
            }
            "PoseidonMdsGate" => {
                vec_ops.extend(eval_unfiltered_base_packed_poseidon_mds::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                ));
            }
            "LowDegreeInterpolationGate" => {
                let subgroup_bits = get_params(gate.id(), "subgroup_bits");
                vec_ops.extend(eval_unfiltered_base_packed_low_degree_interpolation::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    subgroup_bits,
                ));
            }
            "MulExtensionGate" => {
                let num_ops = get_params(gate.id(), "num_ops");
                vec_ops.extend(eval_unfiltered_base_packed_mul_extension::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_ops,
                ));
            }
            "ArithmeticExtensionGate" => {
                let num_ops = get_params(gate.id(), "num_ops");
                vec_ops.extend(eval_unfiltered_base_packed_arithmetic_extension::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_ops,
                ));
            }
            "ReducingExtensionGate" => {
                let num_coeffs = get_params(gate.id(), "num_coeffs");
                vec_ops.extend(eval_unfiltered_base_packed_reducing_extension::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_coeffs,
                ));
            }
            "ReducingGate" => {
                let num_coeffs = get_params(gate.id(), "num_coeffs");
                vec_ops.extend(eval_unfiltered_base_packed_reducing::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    num_coeffs,
                ));
            }
            "CosetInterpolationGate" => {
                let subgroup_bits = get_params(gate.id(), "subgroup_bits");
                let degree = get_params(gate.id(), "degree");
                vec_ops.extend(eval_unfiltered_base_packed_coset_interpolation::<D>(
                    &mut sys.mem,
                    vars_batch,
                    addr_res_batch,
                    subgroup_bits,
                    degree,
                ));
            }
            _ => {
                panic!("Gate type {:?} not found", substring);
            }
        }
    } else {
        panic!("Gate type {:?} not found", gate.id());
    }
    vec_ops.push(VecOpConfig {
        vector_length: vars_batch.len(),
        addr_input_0: addr_filters,
        addr_input_1: addr_res_batch,
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });

    sys.mem.free("sub");
    sys.mem.free("filters");

    let _duration = _start.elapsed();
    // debug!(
    //     "Time elapsed in eval_filtered_base_batch() is: {:?}",
    //     duration
    // );
    vec_ops
}

fn eval_unfiltered_base_packed_arithmetic(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let addr_0 = mem.alloc("arithmetic_0", vars_base.len() * SIZE_F).unwrap();
    let addr_1 = mem.alloc("arithmetic_1", vars_base.len() * SIZE_F).unwrap();

    for i in 0..num_ops {
        // multiplicand_0 * multiplicand_1
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: vars_base.addr_local_wires + 4 * i * vars_base.len() * SIZE_F,
            addr_input_1: vars_base.addr_local_wires + (4 * i + 1) * vars_base.len() * SIZE_F,
            addr_output: addr_0,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // _ * const_0
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_0,
            addr_input_1: vars_base.addr_local_constants,
            addr_output: addr_0,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // addend * const_1
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: vars_base.addr_local_wires + (4 * i + 2) * vars_base.len() * SIZE_F,
            addr_input_1: vars_base.addr_local_constants + vars_base.len() * SIZE_F,
            addr_output: addr_1,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // _ + _
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_0,
            addr_input_1: addr_1,
            addr_output: addr_0,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // output - computed_output
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: vars_base.addr_local_wires + (4 * i + 3) * vars_base.len() * SIZE_F,
            addr_input_1: addr_0,
            addr_output: addr_res_batch + i * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
    }
    mem.free("arithmetic_0");
    mem.free("arithmetic_1");

    vec_ops
}

fn eval_unfiltered_base_packed_base_sum<const B: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    num_limbs: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let addr_sum = vars_base.addr_local_wires;
    let addr_limbs = vars_base.addr_local_wires + SIZE_F;

    let addr_computed_sum = mem.alloc("computed_sum", vars_base.len() * SIZE_F).unwrap();
    for i in 0..num_limbs {
        // sum * alpha
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: if i != 0 { addr_computed_sum } else { 0 },
            addr_input_1: 0, // const B
            addr_output: addr_computed_sum,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        // _ + term
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_computed_sum,
            addr_input_1: addr_limbs + i * vars_base.len() * SIZE_F,
            addr_output: addr_computed_sum,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
    }
    // computed_sum - sum
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_computed_sum,
        addr_input_1: addr_sum,
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    mem.free("computed_sum");

    let addr_sum_constraints_0 = mem
        .alloc("sum_constraints_0", vars_base.len() * SIZE_F)
        .unwrap();
    for i in 0..num_limbs {
        let addr_limb_res = addr_res_batch + (i + 1) * vars_base.len() * SIZE_F;
        for _ in 0..B {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_limbs + i * vars_base.len() * SIZE_F,
                addr_input_1: 0, // const j
                addr_output: addr_sum_constraints_0,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_sum_constraints_0,
                addr_input_1: addr_limb_res,
                addr_output: addr_limb_res,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: true,
            });
        }
    }
    mem.free("sum_constraints_0");

    vec_ops
}

fn sbox_ops(
    mem: &mut MemAlloc,
    addr_state: usize,
    length: usize,
    batch_size: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();
    let addr_sbox_tmp0 = mem.alloc("sbox_0", length * batch_size * SIZE_F).unwrap();
    let addr_sbox_tmp1 = mem.alloc("sbox_1", length * batch_size * SIZE_F).unwrap();
    vec_ops.push(VecOpConfig {
        vector_length: length * batch_size,
        addr_input_0: addr_state,
        addr_input_1: addr_state,
        addr_output: addr_sbox_tmp0,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    vec_ops.push(VecOpConfig {
        vector_length: length * batch_size,
        addr_input_0: addr_sbox_tmp0,
        addr_input_1: addr_sbox_tmp0,
        addr_output: addr_sbox_tmp1,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    vec_ops.push(VecOpConfig {
        vector_length: length * batch_size,
        addr_input_0: addr_state,
        addr_input_1: addr_sbox_tmp0,
        addr_output: addr_state,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    vec_ops.push(VecOpConfig {
        vector_length: length * batch_size,
        addr_input_0: addr_state,
        addr_input_1: addr_sbox_tmp1,
        addr_output: addr_state,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    mem.free("sbox_0");
    mem.free("sbox_1");
    vec_ops
}

fn mds_ops(mem: &mut MemAlloc, addr_state: usize, batch_size: usize) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();
    let addr_mds = mem
        .alloc("mds", SPONGE_WIDTH * batch_size * SIZE_F)
        .unwrap();
    for i in 0..SPONGE_WIDTH {
        vec_ops.push(VecOpConfig {
            vector_length: batch_size,
            addr_input_0: addr_state,
            addr_input_1: 0,
            addr_output: addr_mds + i * SPONGE_WIDTH * SIZE_F,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: batch_size,
            addr_input_0: addr_mds + i * SPONGE_WIDTH * SIZE_F,
            addr_input_1: 0,
            addr_output: addr_state,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
    }
    mem.free("mds");
    vec_ops
}

fn const_ops(addr_state: usize, batch_size: usize) -> VecOpConfig {
    VecOpConfig {
        vector_length: SPONGE_WIDTH * batch_size,
        addr_input_0: addr_state,
        addr_input_1: 0,
        addr_output: addr_state,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VS,
        is_final_output: true,
    }
}

fn eval_unfiltered_base_poseidon(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
) -> Vec<VecOpConfig> {
    const WIRE_SWAP: usize = 2 * SPONGE_WIDTH;
    const START_DELTA: usize = 2 * SPONGE_WIDTH + 1;
    const HALF_N_FULL_ROUNDS: usize = 4;
    const N_PARTIAL_ROUNDS: usize = 22;
    const START_FULL_0: usize = START_DELTA + 4;
    const START_FULL_1: usize = START_PARTIAL + N_PARTIAL_ROUNDS;
    const START_PARTIAL: usize = START_FULL_0 + SPONGE_WIDTH * (HALF_N_FULL_ROUNDS - 1);
    let mut vec_ops = Vec::new();

    let addr_local_wires = vars_base.addr_local_wires;
    let mut addr_res = addr_res_batch;
    let addr_swap = addr_local_wires + WIRE_SWAP * vars_base.len() * SIZE_F;
    let addr_swap_m1 = mem.alloc("swap_m1", SIZE_F * vars_base.len()).unwrap();
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_swap,
        addr_input_1: 0,
        addr_output: addr_swap_m1,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_swap_m1,
        addr_input_1: addr_swap,
        addr_output: addr_res,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res += vars_base.len() * SIZE_F;
    mem.free("swap_m1");
    for i in 0..4 {
        let input_lhs = addr_local_wires + i * vars_base.len() * SIZE_F;
        let input_rhs = addr_local_wires + (i + 4) * vars_base.len() * SIZE_F;
        let delta_i = addr_local_wires + (START_DELTA + i) * vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: input_rhs,
            addr_input_1: input_lhs,
            addr_output: addr_res,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        // swap * _
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_swap,
            addr_input_1: addr_res,
            addr_output: addr_res,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        // _ - delta_i
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_res,
            addr_input_1: delta_i,
            addr_output: addr_res,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res += vars_base.len() * SIZE_F;
    }

    let addr_state = mem
        .alloc("state", vars_base.len() * SPONGE_WIDTH * SIZE_F)
        .unwrap();
    for i in 0..4 {
        let addr_delta_i = addr_local_wires + (START_DELTA + i) * vars_base.len() * SIZE_F;
        let addr_input_lhs = addr_local_wires + i * vars_base.len() * SIZE_F;
        let addr_input_rhs = addr_local_wires + (i + 4) * vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_input_lhs,
            addr_input_1: addr_delta_i,
            addr_output: addr_state + i * vars_base.len() * SIZE_F,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_input_rhs,
            addr_input_1: addr_delta_i,
            addr_output: addr_state + (i + 4) * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
    }

    for r in 0..HALF_N_FULL_ROUNDS {
        vec_ops.push(const_ops(addr_state, vars_base.len()));
        if r != 0 {
            for i in 0..SPONGE_WIDTH {
                let _sbox_in = addr_local_wires
                    + (START_FULL_0 + SPONGE_WIDTH * (r - 1) + i) * vars_base.len() * SIZE_F;
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_state + i * vars_base.len() * SIZE_F,
                    addr_input_1: 0,
                    addr_output: addr_res,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: true,
                });
                addr_res += vars_base.len() * SIZE_F;
            }
        }
        vec_ops.extend(sbox_ops(mem, addr_state, SPONGE_WIDTH, vars_base.len()));
        vec_ops.extend(mds_ops(mem, addr_state, vars_base.len()));
    }

    vec_ops.push(const_ops(addr_state, vars_base.len()));
    vec_ops.extend(mds_ops(mem, addr_state, vars_base.len()));
    for r in 0..(N_PARTIAL_ROUNDS - 1) {
        let sbox_in = addr_local_wires + (START_PARTIAL + r) * vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_state,
            addr_input_1: sbox_in,
            addr_output: addr_res,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res += vars_base.len() * SIZE_F;
        vec_ops.extend(sbox_ops(mem, addr_state, 1, vars_base.len()));
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_state,
            addr_input_1: 0,
            addr_output: addr_state,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.extend(mds_ops(mem, addr_state, vars_base.len()));
    }

    let sbox_in =
        addr_local_wires + (START_PARTIAL + N_PARTIAL_ROUNDS - 1) * vars_base.len() * SIZE_F;
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_state,
        addr_input_1: sbox_in,
        addr_output: addr_res,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res += vars_base.len() * SIZE_F;
    vec_ops.extend(sbox_ops(mem, addr_state, 1, vars_base.len()));
    vec_ops.extend(mds_ops(mem, addr_state, vars_base.len()));

    for r in 0..HALF_N_FULL_ROUNDS {
        vec_ops.push(const_ops(addr_state, vars_base.len()));
        if r != 0 {
            for i in 0..SPONGE_WIDTH {
                let sbox_in = addr_local_wires
                    + (START_FULL_1 + SPONGE_WIDTH * (r - 1) + i) * vars_base.len() * SIZE_F;
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_state + i * vars_base.len() * SIZE_F,
                    addr_input_1: sbox_in,
                    addr_output: addr_res,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: true,
                });
                addr_res += vars_base.len() * SIZE_F;
            }
        }
        vec_ops.extend(sbox_ops(mem, addr_state, SPONGE_WIDTH, vars_base.len()));
        vec_ops.extend(mds_ops(mem, addr_state, vars_base.len()));
    }
    for i in 0..SPONGE_WIDTH {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_state + i * vars_base.len() * SIZE_F,
            addr_input_1: addr_local_wires + (SPONGE_WIDTH + i) * vars_base.len() * SIZE_F,
            addr_output: addr_res,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res += vars_base.len() * SIZE_F;
    }
    mem.free("state");

    vec_ops
}

fn eval_unfiltered_base_packed_constant(
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    num_constraints: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len() * num_constraints,
        addr_input_0: vars_base.addr_local_constants,
        addr_input_1: vars_base.addr_local_wires,
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    vec_ops
}

fn eval_unfiltered_base_packed_exponentiation(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    num_power_bits: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let addr_base = vars_base.addr_local_wires;
    let addr_power_bits = vars_base.addr_local_wires + vars_base.len() * SIZE_F;
    let addr_intermediate_values =
        vars_base.addr_local_wires + (2 + num_power_bits) * vars_base.len() * SIZE_F;
    let addr_output = vars_base.addr_local_wires + (1 + num_power_bits) * vars_base.len() * SIZE_F;

    let addr_prev_intermediate_value = mem
        .alloc("prev_intermediate_value", vars_base.len() * SIZE_F)
        .unwrap();
    let addr_not_cur_bit = mem.alloc("not_cur_bit", vars_base.len() * SIZE_F).unwrap();
    let addr_computed_intermediate_value = mem
        .alloc("computed_intermediate_value", vars_base.len() * SIZE_F)
        .unwrap();
    for i in 0..num_power_bits {
        if i > 0 {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_intermediate_values + (i - 1) * vars_base.len() * SIZE_F,
                addr_input_1: addr_intermediate_values + (i - 1) * vars_base.len() * SIZE_F,
                addr_output: addr_prev_intermediate_value,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }

        let addr_cur_bit = addr_power_bits + (num_power_bits - i - 1) * vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_cur_bit,
            addr_input_1: 0, // P::ONES
            addr_output: addr_not_cur_bit,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });

        // cur_bit * base
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_cur_bit,
            addr_input_1: addr_base,
            addr_output: addr_computed_intermediate_value,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // _ + not_cur_bit
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_computed_intermediate_value,
            addr_input_1: addr_not_cur_bit,
            addr_output: addr_computed_intermediate_value,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // prev_intermediate_value * _
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_prev_intermediate_value,
            addr_input_1: addr_computed_intermediate_value,
            addr_output: addr_computed_intermediate_value,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        // computed_intermediate_value - intermediate_values[i]
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_computed_intermediate_value,
            addr_input_1: addr_intermediate_values + i * vars_base.len() * SIZE_F,
            addr_output: addr_res_batch + i * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
    }
    //output - intermediate_values[self.num_power_bits - 1]
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_output,
        addr_input_1: addr_intermediate_values + (num_power_bits - 1) * vars_base.len() * SIZE_F,
        addr_output: addr_res_batch + num_power_bits * vars_base.len() * SIZE_F,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    mem.free("prev_intermediate_value");
    mem.free("not_cur_bit");
    mem.free("computed_intermediate_value");

    vec_ops
}

fn eval_unfiltered_base_packed_public_input(
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let addr_public_inputs_hash = vars_base.addr_public_inputs_hash;
    for i in 0..4 {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: vars_base.addr_local_wires + i * vars_base.len() * SIZE_F,
            addr_input_1: addr_public_inputs_hash + i * SIZE_F,
            addr_output: addr_res_batch + i * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
            is_final_output: true,
        });
    }

    vec_ops
}

fn eval_unfiltered_base_packed_random_access(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    bits: usize,
    num_copies: usize,
    num_extra_constants: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let vec_size = 1 << bits;
    let num_routed_wires = (2 + vec_size) * num_copies + num_extra_constants;

    let mut arb_offset = 0;
    for copy in 0..num_copies {
        let addr_access_index =
            vars_base.addr_local_wires + (2 + vec_size) * copy * vars_base.len() * SIZE_F;
        let addr_list_items =
            vars_base.addr_local_wires + ((2 + vec_size) * copy + 2) * vars_base.len() * SIZE_F;
        let addr_claimed_element =
            vars_base.addr_local_wires + ((2 + vec_size) * copy + 1) * vars_base.len() * SIZE_F;
        let addr_bits = vars_base.addr_local_wires
            + (num_routed_wires + copy * bits) * vars_base.len() * SIZE_F;

        for b in 0..bits {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_bits + b * vars_base.len() * SIZE_F,
                addr_input_1: 0, // const b
                addr_output: addr_res_batch + arb_offset * vars_base.len() * SIZE_F,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VS,
                is_final_output: true,
            });
            arb_offset += 1;
        }

        let addr_acc = mem.alloc("acc", vars_base.len() * SIZE_F).unwrap();
        for b in (0..bits).rev() {
            // acc + acc
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: if b != 0 { addr_acc } else { 0 },
                addr_input_1: if b != 0 { addr_acc } else { 0 },
                addr_output: addr_acc,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            // _ + b
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_acc,
                addr_input_1: addr_bits + b * vars_base.len() * SIZE_F,
                addr_output: addr_acc,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }
        // reconstructed_index - access_index
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_acc,
            addr_input_1: addr_access_index,
            addr_output: addr_res_batch + arb_offset * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        arb_offset += 1;
        mem.free("acc");

        let addr_list_items_tmp = mem
            .alloc("list_items_tmp", vars_base.len() * SIZE_F)
            .unwrap();
        for b in 0..bits {
            for i in 0..(1 << (bits - b - 1)) {
                // y - x
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_list_items + (i * 2 + 1) * vars_base.len() * SIZE_F,
                    addr_input_1: addr_list_items + i * 2 * vars_base.len() * SIZE_F,
                    addr_output: addr_list_items_tmp,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                // b * _
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_list_items_tmp,
                    addr_input_1: 0, // const
                    addr_output: addr_list_items_tmp,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                // x + _
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_list_items + i * 2 * vars_base.len() * SIZE_F,
                    addr_input_1: addr_list_items_tmp,
                    addr_output: addr_list_items + i * vars_base.len() * SIZE_F,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
            }
        }
        mem.free("list_items_tmp");

        // list_items[0] - claimed_element
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_list_items,
            addr_input_1: addr_claimed_element,
            addr_output: addr_res_batch + arb_offset * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        arb_offset += 1;
    }
    for i in 0..num_extra_constants {
        // vars.local_constants[i] - vars.local_wires[self.wire_extra_constant(i)]
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: vars_base.addr_local_constants + i * vars_base.len() * SIZE_F,
            addr_input_1: vars_base.addr_local_wires
                + ((2 + vec_size) * num_copies + i) * vars_base.len() * SIZE_F,
            addr_output: addr_res_batch + arb_offset * vars_base.len() * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        arb_offset += 1;
    }

    vec_ops
}

fn eval_unfiltered_base_packed_u32_add_many(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_addends: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let wire_ith_op_jth_addend = |i: usize, j: usize| {
        vars_base.addr_local_wires + ((num_addends + 3) * i + j) * vars_base.len() * SIZE_F
    };
    let wire_ith_carry = |i: usize| {
        vars_base.addr_local_wires
            + ((num_addends + 3) * i + num_addends) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_result = |i: usize| {
        vars_base.addr_local_wires
            + ((num_addends + 3) * i + num_addends + 1) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_carry = |i: usize| {
        vars_base.addr_local_wires
            + ((num_addends + 3) * i + num_addends + 2) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_jth_limb = |i: usize, j: usize| {
        vars_base.addr_local_wires
            + ((num_addends + 3) * num_ops + 18 * i + j) * vars_base.len() * SIZE_F
    };
    let mut vec_ops = Vec::new();
    for i in 0..num_ops {
        let addr_computed_output = mem
            .alloc("computed_output", vars_base.len() * SIZE_F)
            .unwrap();
        for j in 0..num_addends {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: wire_ith_op_jth_addend(i, j),
                addr_input_1: if j != 0 { addr_computed_output } else { 0 },
                addr_output: addr_computed_output,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_carry(i),
            addr_input_1: addr_computed_output,
            addr_output: addr_computed_output,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });

        let addr_combined_output = mem
            .alloc("combined_output", vars_base.len() * SIZE_F)
            .unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_output_carry(i),
            addr_input_1: 0, //const,
            addr_output: addr_combined_output,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_computed_output,
            addr_input_1: wire_ith_output_result(i),
            addr_output: addr_combined_output,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_output,
            addr_input_1: addr_computed_output,
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        mem.free("computed_output");
        mem.free("combined_output");

        let addr_combined_result_limbs = mem
            .alloc("combined_result_limbs", vars_base.len() * SIZE_F)
            .unwrap();
        let addr_combined_carry_limbs = mem
            .alloc("combined_carry_limbs", vars_base.len() * SIZE_F)
            .unwrap();

        for j in (0..18).rev() {
            let max_limb = 1 << 2;
            let addr_this_minus = mem.alloc("this_minus", vars_base.len() * SIZE_F).unwrap();
            for x in 0..max_limb {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j),
                    addr_input_1: 0, //const x
                    addr_output: addr_this_minus,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_this_minus,
                    addr_input_1: addr_res_batch,
                    addr_output: addr_res_batch,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: x == max_limb - 1,
                });
            }
            mem.free("this_minus");
            addr_res_batch += vars_base.len() * SIZE_F;

            if j < 16 {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: 0, // const base
                    addr_input_1: if j != 0 {
                        addr_combined_result_limbs
                    } else {
                        0
                    },
                    addr_output: addr_combined_result_limbs,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j),
                    addr_input_1: addr_combined_result_limbs,
                    addr_output: addr_combined_result_limbs,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
            } else {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: 0, // const base
                    addr_input_1: if j != 16 {
                        addr_combined_carry_limbs
                    } else {
                        0
                    },
                    addr_output: addr_combined_carry_limbs,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j),
                    addr_input_1: addr_combined_carry_limbs,
                    addr_output: addr_combined_carry_limbs,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
            }
        }
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_result_limbs,
            addr_input_1: wire_ith_output_result(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_carry_limbs,
            addr_input_1: wire_ith_output_carry(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        mem.free("combined_result_limbs");
        mem.free("combined_carry_limbs");
    }
    vec_ops
}

fn eval_unfiltered_base_packed_u32_range_check(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_input_limbs: usize,
) -> Vec<VecOpConfig> {
    let aux_limbs_per_input_limb = 16;
    let wire_ith_input_limb_jth_aux_limb = |i: usize, j: usize| {
        vars_base.addr_local_wires
            + (num_input_limbs + aux_limbs_per_input_limb * i + j) * vars_base.len() * SIZE_F
    };
    let wire_ith_input_limb = |i: usize| vars_base.addr_local_wires + i * vars_base.len() * SIZE_F;

    let mut vec_ops = Vec::new();
    for i in 0..num_input_limbs {
        let addr_computed_sum = mem.alloc("computed_sum", vars_base.len() * SIZE_F).unwrap();
        for j in 0..aux_limbs_per_input_limb {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: wire_ith_input_limb_jth_aux_limb(i, j),
                addr_input_1: if j != 0 { addr_computed_sum } else { 0 },
                addr_output: addr_computed_sum,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_computed_sum,
                addr_input_1: 0, // base
                addr_output: addr_computed_sum,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_computed_sum,
            addr_input_1: wire_ith_input_limb(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        for j in 0..16 {
            let addr_aux_limb_minus = mem
                .alloc("aux_limb_minus", vars_base.len() * SIZE_F)
                .unwrap();
            for k in 0..4 {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_input_limb_jth_aux_limb(i, j),
                    addr_input_1: 0, // const k
                    addr_output: addr_aux_limb_minus,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_aux_limb_minus,
                    addr_input_1: addr_res_batch,
                    addr_output: addr_res_batch,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: k == 3,
                });
            }
            addr_res_batch += vars_base.len() * SIZE_F;
            mem.free("aux_limb_minus");
        }

        mem.free("computed_sum");
    }
    vec_ops
}

fn eval_unfiltered_base_packed_comparison(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_bits: usize,
    num_chunks: usize,
) -> Vec<VecOpConfig> {
    let wire_first_input = || vars_base.addr_local_wires + 0 * vars_base.len() * SIZE_F;
    let wire_second_input = || vars_base.addr_local_wires + 1 * vars_base.len() * SIZE_F;
    let wire_result_bool = || vars_base.addr_local_wires + 2 * vars_base.len() * SIZE_F;
    let wire_most_significant_diff = || vars_base.addr_local_wires + 3 * vars_base.len() * SIZE_F;

    let wire_first_chunk_val =
        |chunk: usize| vars_base.addr_local_wires + (4 + chunk) * vars_base.len() * SIZE_F;
    let wire_second_chunk_val = |chunk: usize| {
        vars_base.addr_local_wires + (4 + num_chunks + chunk) * vars_base.len() * SIZE_F
    };
    let wire_equality_dummy = |chunk: usize| {
        vars_base.addr_local_wires + (4 + 2 * num_chunks + chunk) * vars_base.len() * SIZE_F
    };
    let wire_chunks_equal = |chunk: usize| {
        vars_base.addr_local_wires + (4 + 3 * num_chunks + chunk) * vars_base.len() * SIZE_F
    };
    let wire_intermediate_value = |chunk: usize| {
        vars_base.addr_local_wires + (4 + 4 * num_chunks + chunk) * vars_base.len() * SIZE_F
    };
    let wire_most_significant_diff_bit = |bit_index: usize| {
        vars_base.addr_local_wires + (4 + 5 * num_chunks + bit_index) * vars_base.len() * SIZE_F
    };

    let mut vec_ops = Vec::new();
    let addr_first_chunks_combined = mem
        .alloc("first_chunks_combined", vars_base.len() * SIZE_F)
        .unwrap();
    let addr_second_chunks_combined = mem
        .alloc("second_chunks_combined", vars_base.len() * SIZE_F)
        .unwrap();
    for i in 0..num_chunks {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // const
            addr_input_1: if i != 0 {
                addr_first_chunks_combined
            } else {
                0
            },
            addr_output: addr_first_chunks_combined,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_first_chunk_val(i),
            addr_input_1: addr_first_chunks_combined,
            addr_output: addr_first_chunks_combined,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
    }
    for i in 0..num_chunks {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // const
            addr_input_1: if i != 0 {
                addr_second_chunks_combined
            } else {
                0
            },
            addr_output: addr_second_chunks_combined,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_second_chunk_val(i),
            addr_input_1: addr_second_chunks_combined,
            addr_output: addr_second_chunks_combined,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
    }
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_first_chunks_combined,
        addr_input_1: wire_first_input(),
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res_batch += vars_base.len() * SIZE_F;
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_second_chunks_combined,
        addr_input_1: wire_second_input(),
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res_batch += vars_base.len() * SIZE_F;
    mem.free("first_chunks_combined");
    mem.free("second_chunks_combined");

    let chunk_size = 1 << ceil_div_usize(num_bits, num_chunks);
    let addr_most_significant_diff_so_far = mem
        .alloc("most_significant_diff_so_far", vars_base.len() * SIZE_F)
        .unwrap();

    for i in 0..num_chunks {
        let addr_first_chunks_minus = mem
            .alloc("first_chunks_minus", vars_base.len() * SIZE_F)
            .unwrap();
        for j in 0..chunk_size {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: wire_first_chunk_val(i),
                addr_input_1: 0, // const
                addr_output: addr_first_chunks_minus,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_first_chunks_minus,
                addr_input_1: addr_res_batch,
                addr_output: addr_res_batch,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: j == chunk_size - 1,
            });
        }
        addr_res_batch += vars_base.len() * SIZE_F;

        for j in 0..chunk_size {
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: wire_second_chunk_val(i),
                addr_input_1: 0, // const
                addr_output: addr_first_chunks_minus,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: addr_first_chunks_minus,
                addr_input_1: addr_res_batch,
                addr_output: addr_res_batch,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: j == chunk_size - 1,
            });
        }
        mem.free("first_chunks_minus");

        let addr_difference = mem.alloc("difference", vars_base.len() * SIZE_F).unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_first_chunk_val(i),
            addr_input_1: wire_second_chunk_val(i),
            addr_output: addr_difference,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_difference,
            addr_input_1: wire_equality_dummy(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_res_batch,
            addr_input_1: wire_chunks_equal(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_res_batch,
            addr_input_1: 0, // const
            addr_output: addr_res_batch,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_chunks_equal(i),
            addr_input_1: addr_difference,
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_chunks_equal(i),
            addr_input_1: if i != 0 {
                addr_most_significant_diff_so_far
            } else {
                0
            },
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_intermediate_value(i),
            addr_input_1: addr_res_batch, // const
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        let addr_one_minus_chunks_equal = mem
            .alloc("one_minus_chunks_equal", vars_base.len() * SIZE_F)
            .unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // const
            addr_input_1: wire_chunks_equal(i),
            addr_output: addr_one_minus_chunks_equal,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_one_minus_chunks_equal,
            addr_input_1: addr_difference,
            addr_output: addr_one_minus_chunks_equal,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_intermediate_value(i),
            addr_input_1: addr_one_minus_chunks_equal,
            addr_output: addr_most_significant_diff_so_far,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        mem.free("one_minus_chunks_equal");
        mem.free("difference");
    }

    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: wire_most_significant_diff(),
        addr_input_1: addr_most_significant_diff_so_far,
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res_batch += vars_base.len() * SIZE_F;

    for i in 0..ceil_div_usize(num_bits, num_chunks) + 1 {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // const
            addr_input_1: wire_most_significant_diff_bit(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_res_batch,
            addr_input_1: wire_most_significant_diff_bit(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
    }

    let addr_bits_combined = mem
        .alloc("bits_combined", vars_base.len() * SIZE_F)
        .unwrap();

    for i in 0..ceil_div_usize(num_bits, num_chunks) + 1 {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // const
            addr_input_1: if i != 0 { addr_bits_combined } else { 0 },
            addr_output: addr_bits_combined,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_most_significant_diff_bit(i),
            addr_input_1: addr_bits_combined,
            addr_output: addr_bits_combined,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
    }
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_most_significant_diff_so_far,
        addr_input_1: 0, // const
        addr_output: addr_res_batch,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
        is_final_output: false,
    });
    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: addr_res_batch,
        addr_input_1: addr_bits_combined,
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });
    addr_res_batch += vars_base.len() * SIZE_F;

    vec_ops.push(VecOpConfig {
        vector_length: vars_base.len(),
        addr_input_0: wire_result_bool(),
        addr_input_1: wire_most_significant_diff_bit(ceil_div_usize(num_bits, num_chunks)),
        addr_output: addr_res_batch,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
        is_final_output: true,
    });

    vec_ops
}
fn eval_unfiltered_base_packed_u32_arithmetic(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let routed_wires_per_op = || 6;
    let wire_ith_multiplicand_0 = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 0) * vars_base.len() * SIZE_F
    };
    let wire_ith_multiplicand_1 = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 1) * vars_base.len() * SIZE_F
    };
    let wire_ith_addend = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 2) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_low_half = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 3) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_high_half = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 4) * vars_base.len() * SIZE_F
    };
    let wire_ith_inverse = |i: usize| {
        vars_base.addr_local_wires + (routed_wires_per_op() * i + 5) * vars_base.len() * SIZE_F
    };
    let wire_ith_output_jth_limb = |i: usize, j: usize| {
        assert!(i < num_ops);
        assert!(j < 32);
        vars_base.addr_local_wires
            + (routed_wires_per_op() * num_ops + 32 * i + j) * vars_base.len() * SIZE_F
    };

    let mut vec_ops = Vec::new();
    for i in 0..num_ops {
        let addr_computed_output = mem
            .alloc("computed_output", vars_base.len() * SIZE_F)
            .unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_multiplicand_0(i),
            addr_input_1: wire_ith_multiplicand_1(i),
            addr_output: addr_computed_output,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_addend(i),
            addr_input_1: 0, // _
            addr_output: addr_computed_output,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });

        let addr_combined_output = mem
            .alloc("combined_output", vars_base.len() * SIZE_F)
            .unwrap();
        let addr_diff = mem.alloc("diff", vars_base.len() * SIZE_F).unwrap();
        let addr_hi_not_max = mem.alloc("hi_not_max", vars_base.len() * SIZE_F).unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0, // u32_max
            addr_input_1: wire_ith_output_high_half(i),
            addr_output: addr_diff,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_inverse(i),
            addr_input_1: addr_diff,
            addr_output: addr_hi_not_max,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_hi_not_max,
            addr_input_1: 0, // const
            addr_output: addr_hi_not_max,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_hi_not_max,
            addr_input_1: wire_ith_output_low_half(i),
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        mem.free("diff");
        mem.free("hi_not_max");

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_output_high_half(i), // output_high
            addr_input_1: 0,                            // base
            addr_output: addr_combined_output,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_output,
            addr_input_1: wire_ith_output_low_half(i), // output_low
            addr_output: addr_combined_output,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_output,
            addr_input_1: addr_computed_output,
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        mem.free("computed_output");
        mem.free("combined_output");

        let addr_combined_low_limbs = mem
            .alloc("combined_low_limbs", vars_base.len() * SIZE_F)
            .unwrap();
        let addr_combined_high_limbs = mem
            .alloc("combined_high_limbs", vars_base.len() * SIZE_F)
            .unwrap();
        let midpoint = 32 / 2;
        for j in (0..32).rev() {
            let max_limb = 4;
            let addr_this_limb_minus = mem
                .alloc("this_limb_minus", vars_base.len() * SIZE_F)
                .unwrap();
            for k in 0..max_limb {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j), // this_limb
                    addr_input_1: 0,                              // const
                    addr_output: addr_this_limb_minus,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_this_limb_minus,
                    addr_input_1: addr_res_batch,
                    addr_output: addr_res_batch,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: k == max_limb - 1,
                });
            }
            mem.free("this_limb_minus");
            addr_res_batch += vars_base.len() * SIZE_F;

            if j < midpoint {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: 0, // const base
                    addr_input_1: if j != 0 { addr_combined_low_limbs } else { 0 },
                    addr_output: addr_combined_low_limbs,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j), // this_limb
                    addr_input_1: addr_combined_low_limbs,
                    addr_output: addr_combined_low_limbs,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
            } else {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: 0, // const base
                    addr_input_1: if j != midpoint {
                        addr_combined_high_limbs
                    } else {
                        0
                    },
                    addr_output: addr_combined_high_limbs,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j), // this_limb
                    addr_input_1: addr_combined_high_limbs,
                    addr_output: addr_combined_high_limbs,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
            }
        }
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_low_limbs,
            addr_input_1: wire_ith_output_low_half(i), // output_low
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_high_limbs,
            addr_input_1: wire_ith_output_high_half(i), // output_high
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
        mem.free("combined_low_limbs");
        mem.free("combined_high_limbs");
    }

    vec_ops
}

fn eval_unfiltered_base_packed_u32_subtraction(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let wire_ith_input_x =
        |i: usize| vars_base.addr_local_wires + (5 * i) * vars_base.len() * SIZE_F;
    let wire_ith_input_y =
        |i: usize| vars_base.addr_local_wires + (5 * i + 1) * vars_base.len() * SIZE_F;
    let wire_ith_input_borrow =
        |i: usize| vars_base.addr_local_wires + (5 * i + 2) * vars_base.len() * SIZE_F;
    let wire_ith_output_result =
        |i: usize| vars_base.addr_local_wires + (5 * i + 3) * vars_base.len() * SIZE_F;
    let wire_ith_output_borrow =
        |i: usize| vars_base.addr_local_wires + (5 * i + 4) * vars_base.len() * SIZE_F;
    let wire_ith_output_jth_limb = |i: usize, j: usize| {
        debug_assert!(i < num_ops);
        debug_assert!(j < 16);
        vars_base.addr_local_wires + (5 * num_ops + 16 * i + j) * vars_base.len() * SIZE_F
    };

    let mut vec_ops = Vec::new();
    for i in 0..num_ops {
        let addr_result_initial = mem
            .alloc("result_initial", vars_base.len() * SIZE_F)
            .unwrap();

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_input_x(i), // input_x
            addr_input_1: wire_ith_input_y(i), // input_y
            addr_output: addr_result_initial,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_result_initial,
            addr_input_1: wire_ith_input_borrow(i), // input_borrow
            addr_output: addr_result_initial,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_output_borrow(i), // output_borrow
            addr_input_1: 0,                         // base
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_result_initial,
            addr_input_1: addr_res_batch,
            addr_output: addr_res_batch,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_output_result(i), // output_result
            addr_input_1: addr_res_batch,
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
        mem.free("result_initial");

        let addr_combined_limbs = mem
            .alloc("combined_limbs", vars_base.len() * SIZE_F)
            .unwrap();
        for j in (0..16).rev() {
            let addr_this_limb_minus = mem
                .alloc("this_limb_minus", vars_base.len() * SIZE_F)
                .unwrap();
            let max_limb = 1 << 2;
            for k in 0..max_limb {
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: wire_ith_output_jth_limb(i, j), // this_limb
                    addr_input_1: 0,                              // const k
                    addr_output: addr_this_limb_minus,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                    is_final_output: false,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: vars_base.len(),
                    addr_input_0: addr_this_limb_minus,
                    addr_input_1: addr_res_batch,
                    addr_output: addr_res_batch,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                    is_final_output: k == max_limb - 1,
                });
            }
            addr_res_batch += vars_base.len() * SIZE_F;

            // combined_limbs = combined_limbs * limb_base + this_limb;
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: 0, // limb_base
                addr_input_1: if j != 15 { addr_combined_limbs } else { 0 },
                addr_output: addr_combined_limbs,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: vars_base.len(),
                addr_input_0: wire_ith_output_jth_limb(i, j), // this_limb
                addr_input_1: addr_combined_limbs,
                addr_output: addr_combined_limbs,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });

            mem.free("this_limb_minus");
        }

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_combined_limbs,
            addr_input_1: wire_ith_output_result(i), // output_result
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;

        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: 0,                         // ONES
            addr_input_1: wire_ith_output_borrow(i), // output_borrow
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: wire_ith_output_borrow(i), // output_borrow
            addr_input_1: addr_res_batch,
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
        mem.free("combined_limbs");
    }
    vec_ops
}

fn eval_unfiltered_base_packed_poseidon_mds<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
) -> Vec<VecOpConfig> {
    let wires_input = |i: usize| {
        assert!(i < SPONGE_WIDTH);
        vars_base.addr_local_wires + i * D * vars_base.len() * SIZE_F
    };
    let wires_output = |i: usize| {
        assert!(i < SPONGE_WIDTH);
        addr_res_batch + (SPONGE_WIDTH + i) * D * vars_base.len() * SIZE_F
    };
    let mut vec_ops = Vec::new();
    let addr_computed_outputs = mem
        .alloc(
            "computed_outputs",
            SPONGE_WIDTH * vars_base.len() * SIZE_F * D,
        )
        .unwrap();
    for i in 0..SPONGE_WIDTH {
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            SPONGE_WIDTH * vars_base.len(),
            wires_input(i),
            if i != 0 { addr_computed_outputs } else { 0 },
            addr_computed_outputs,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            SPONGE_WIDTH * vars_base.len(),
            wires_input(i),
            addr_computed_outputs,
            addr_computed_outputs,
            VecOpSrc::VV,
            false,
        ));
    }
    vec_ops.extend(VecOpExtension::<D>::sub(
        vars_base.len() * SPONGE_WIDTH,
        wires_output(0),
        addr_computed_outputs,
        addr_res_batch,
        VecOpSrc::VV,
        true,
    ));

    vec_ops
}

fn eval_unfiltered_base_packed_low_degree_interpolation<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    subgroup_bits: usize,
) -> Vec<VecOpConfig> {
    let num_points = 1 << subgroup_bits;
    let start_values = || 1;
    let start_evaluation_point = || start_values() + num_points * D;
    let start_evaluation_value = || start_evaluation_point() + D;
    let start_coeffs = || start_evaluation_value() + D;
    let end_coeffs = || start_coeffs() + D * num_points;
    let powers_shift_fn = |i: usize| {
        assert!(0 < i && i < num_points);
        if i == 1 {
            return vars_base.addr_local_wires;
        }
        vars_base.addr_local_wires + (end_coeffs() + i - 2) * vars_base.len() * SIZE_F
    };
    let wires_coeff = |i: usize| {
        assert!(i < num_points);
        vars_base.addr_local_wires + (start_coeffs() + i * D) * vars_base.len() * SIZE_F
    };
    let powers_evaluation_point = |i: usize| {
        assert!(0 < i && i < num_points);
        if i == 1 {
            return vars_base.addr_local_wires
                + start_evaluation_point() * vars_base.len() * SIZE_F;
        }
        vars_base.addr_local_wires
            + (end_coeffs() + num_points - 2 + (i - 2) * D) * vars_base.len() * SIZE_F
    };
    let wires_evaluation_value =
        || vars_base.addr_local_wires + start_evaluation_value() * vars_base.len() * SIZE_F;

    let mut vec_ops = Vec::new();

    let coeffs = (0..num_points).map(|i| wires_coeff(i)).collect::<Vec<_>>();
    let powers_shift = (1..num_points)
        .map(|i| powers_shift_fn(i))
        .collect::<Vec<_>>();
    for i in 1..num_points - 1 {
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: powers_shift[i - 1], // powers_shift
            addr_input_1: powers_shift[0],     // shift
            addr_output: addr_res_batch,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: vars_base.len(),
            addr_input_0: addr_res_batch,
            addr_input_1: powers_shift[i], // powers_shift
            addr_output: addr_res_batch,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        addr_res_batch += vars_base.len() * SIZE_F;
    }

    let addr_altered_interpolant = mem
        .alloc(
            "altered_interpolant",
            num_points * vars_base.len() * SIZE_F * D,
        )
        .unwrap();
    for i in 0..num_points {
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            vars_base.len(),
            powers_shift[i], // powers_shift
            coeffs[i],       // coeffs
            addr_altered_interpolant + i * vars_base.len() * SIZE_F * D,
            VecOpSrc::VV,
            false,
        ));
    }
    for _i in 0..num_points {
        for j in 0..num_points {
            vec_ops.extend(VecOpExtension::<D>::scalar_mul(
                vars_base.len(),
                addr_altered_interpolant + j * vars_base.len() * SIZE_F * D,
                0, // const
                addr_res_batch,
                VecOpSrc::VV,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::add(
                vars_base.len(),
                addr_res_batch,
                0, // coeff
                addr_res_batch,
                VecOpSrc::VV,
                false,
            ));
        }
        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            0, // value
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            true,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;
    }
    mem.free("altered_interpolant");

    let evaluation_point_powers = (1..num_points)
        .map(|i| powers_evaluation_point(i))
        .collect::<Vec<_>>();
    for i in 1..num_points - 1 {
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            evaluation_point_powers[i - 1], //evaluation_point_powers
            evaluation_point_powers[0],     // evaluation_point
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            addr_res_batch,
            evaluation_point_powers[i], //evaluation_point_powers
            addr_res_batch,
            VecOpSrc::VV,
            true,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;
    }

    let addr_eval_with_powers = mem
        .alloc("eval_with_powers", vars_base.len() * SIZE_F * D)
        .unwrap();
    for i in 1..num_points {
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            coeffs[i],                      // interpolant
            evaluation_point_powers[i - 1], // evaluation_point_powers
            addr_eval_with_powers,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            vars_base.len(),
            addr_eval_with_powers,
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
    }
    mem.free("eval_with_powers");
    vec_ops.extend(VecOpExtension::<D>::sub(
        vars_base.len(),
        wires_evaluation_value(), //evaluation_value
        addr_res_batch,
        addr_res_batch,
        VecOpSrc::VV,
        true,
    ));

    vec_ops
}

fn eval_unfiltered_base_packed_reducing_extension<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_coeffs: usize,
) -> Vec<VecOpConfig> {
    let wires_output = || vars_base.addr_local_wires;
    let wires_alpha = || vars_base.addr_local_wires + D * vars_base.len() * SIZE_F;
    let wires_old_acc = || vars_base.addr_local_wires + 2 * D * vars_base.len() * SIZE_F;
    let wires_coeff =
        |i: usize| vars_base.addr_local_wires + (3 + i) * D * vars_base.len() * SIZE_F;
    let start_accs = || (3 + num_coeffs) * D;
    let wires_accs = |i: usize| {
        assert!(i < num_coeffs);
        if i == num_coeffs - 1 {
            return wires_output();
        }
        vars_base.addr_local_wires + (start_accs() + i * D) * vars_base.len() * SIZE_F
    };

    let mut vec_ops = Vec::new();

    for i in 0..num_coeffs {
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            if i == 0 {
                wires_old_acc()
            } else {
                wires_accs(i - 1)
            }, // coeffs
            wires_alpha(),
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            vars_base.len(),
            addr_res_batch,
            wires_coeff(i), // coeffs
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            wires_accs(i), // accs
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            true,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;
    }

    vec_ops
}

fn eval_unfiltered_base_packed_reducing<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    addr_res_batch: usize,
    num_coeffs: usize,
) -> Vec<VecOpConfig> {
    eval_unfiltered_base_packed_reducing_extension::<D>(mem, vars_base, addr_res_batch, num_coeffs)
}

fn eval_unfiltered_base_packed_mul_extension<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let wires_ith_multiplicand_0 =
        |i: usize| vars_base.addr_local_wires + (3 * D * i) * vars_base.len() * SIZE_F;
    let wires_ith_multiplicand_1 =
        |i: usize| vars_base.addr_local_wires + (3 * D * i + D) * vars_base.len() * SIZE_F;
    let wires_ith_output =
        |i: usize| vars_base.addr_local_wires + (3 * D * i + 2 * D) * vars_base.len() * SIZE_F;
    let mut vec_ops = Vec::new();

    for i in 0..num_ops {
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            wires_ith_multiplicand_0(i), // multiplicand_0
            wires_ith_multiplicand_1(i), // multiplicand_1
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            vars_base.len(),
            vars_base.addr_local_constants, // const
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            wires_ith_output(i), // output
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            true,
        ));

        addr_res_batch += vars_base.len() * SIZE_F * D;
    }
    vec_ops
}

fn eval_unfiltered_base_packed_arithmetic_extension<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    num_ops: usize,
) -> Vec<VecOpConfig> {
    let wires_ith_multiplicand_0 =
        |i: usize| vars_base.addr_local_wires + (4 * D * i) * vars_base.len() * SIZE_F;
    let wires_ith_multiplicand_1 =
        |i: usize| vars_base.addr_local_wires + (4 * D * i + D) * vars_base.len() * SIZE_F;
    let wires_ith_addend =
        |i: usize| vars_base.addr_local_wires + (4 * D * i + 2 * D) * vars_base.len() * SIZE_F;
    let wires_ith_output =
        |i: usize| vars_base.addr_local_wires + (4 * D * i + 3 * D) * vars_base.len() * SIZE_F;
    let const_0 = vars_base.addr_local_constants;
    let const_1 = vars_base.addr_local_constants + vars_base.len() * SIZE_F;

    let mut vec_ops = Vec::new();
    for i in 0..num_ops {
        let addr_addend = mem.alloc("addend", vars_base.len() * SIZE_F * D).unwrap();
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            wires_ith_multiplicand_0(i), // multiplicand_0
            wires_ith_multiplicand_1(i), // multiplicand_1
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            vars_base.len(),
            addr_res_batch,
            const_0, // const_0
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            vars_base.len(),
            wires_ith_addend(i), // addend
            const_1,             // const_1
            addr_addend,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            vars_base.len(),
            addr_res_batch,
            addr_addend,
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        mem.free("addend");

        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            wires_ith_output(i), // output
            addr_res_batch,
            addr_res_batch,
            VecOpSrc::VV,
            true,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;
    }
    vec_ops
}

fn eval_unfiltered_base_packed_coset_interpolation<const D: usize>(
    mem: &mut MemAlloc,
    vars_base: EvaluationVarsBaseBatch,
    mut addr_res_batch: usize,
    subgroup_bits: usize,
    degree: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let start_values = || 1;
    let num_points = || 1 << subgroup_bits;
    let start_evaluation_point = || start_values() + num_points() * D;
    let wires_evaluation_point = || start_evaluation_point();
    let start_evaluation_value = || start_evaluation_point() + D;
    let start_intermediates = || start_evaluation_value() + D;
    let num_intermediates = || (num_points() - 2) / (degree - 1);
    let wires_shifted_evaluation_point = || start_intermediates() + D * 2 * num_intermediates();
    let wires_value = |i: usize| {
        assert!(i < num_points());
        start_values() + i * D
    };
    let wires_evaluation_value = || start_evaluation_value();

    let num_intermediates = || (num_points() - 2) / (degree - 1);
    let wires_intermediate_eval = |i: usize| {
        debug_assert!(i < num_intermediates());
        start_intermediates() + i * D
    };
    let wires_intermediate_prod = |i: usize| {
        debug_assert!(i < num_intermediates());
        start_intermediates() + num_intermediates() * D + i * D
    };

    let addr_shift = vars_base.addr_local_constants;
    let addr_evaluation_point =
        vars_base.addr_local_constants + wires_evaluation_point() * vars_base.len() * SIZE_F;
    let addr_shifted_evaluation_point = vars_base.addr_local_constants
        + wires_shifted_evaluation_point() * vars_base.len() * SIZE_F;
    vec_ops.extend(VecOpExtension::<D>::scalar_mul(
        vars_base.len(),
        addr_shifted_evaluation_point,
        addr_shift,
        addr_res_batch,
        VecOpSrc::VV,
        false,
    ));
    vec_ops.extend(VecOpExtension::<D>::sub(
        vars_base.len(),
        addr_evaluation_point,
        addr_res_batch,
        addr_res_batch,
        VecOpSrc::VV,
        false,
    ));
    addr_res_batch += vars_base.len() * SIZE_F * D;

    let addr_domain = mem.alloc("domain", (1 << subgroup_bits) * SIZE_F).unwrap();
    let values = (0..num_points())
        .map(|i| vars_base.addr_local_wires + wires_value(i) * vars_base.len() * SIZE_F)
        .collect::<Vec<_>>();

    let addr_weighted_values = (0..degree)
        .map(|i| {
            mem.alloc(
                format!("weighted_values_{}", i).as_str(),
                vars_base.len() * SIZE_F * D,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    for i in 0..degree {
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            vars_base.len(),
            values[i],
            0,
            addr_weighted_values[i],
            VecOpSrc::VV,
            false,
        ));
    }

    let addr_computed_eval = mem
        .alloc("computed_eval", vars_base.len() * SIZE_F * D)
        .unwrap();
    let addr_computed_prod = mem
        .alloc("computed_prod", vars_base.len() * SIZE_F * D)
        .unwrap();
    let addr_eval = mem.alloc("eval", vars_base.len() * SIZE_F * D).unwrap();
    let addr_term = mem.alloc("term", vars_base.len() * SIZE_F * D).unwrap();
    for i in 0..degree {
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            1,
            0,
            addr_domain + i * SIZE_F,
            addr_term,
            VecOpSrc::VS,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::sub(
            vars_base.len(),
            addr_shifted_evaluation_point,
            addr_term,
            addr_term,
            VecOpSrc::VS,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            addr_weighted_values[i],
            if i != 0 { addr_computed_prod } else { 0 },
            addr_eval,
            VecOpSrc::VV,
            false,
        ));
        // eval * term
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            if i != 0 { addr_computed_eval } else { 0 },
            addr_term,
            addr_computed_eval,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            vars_base.len(),
            addr_eval,
            addr_computed_eval,
            addr_computed_eval,
            VecOpSrc::VV,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            addr_computed_prod,
            addr_term,
            addr_computed_prod,
            VecOpSrc::VV,
            false,
        ));
    }

    for i in 0..num_intermediates() {
        let addr_intermediate_eval =
            vars_base.addr_local_wires + wires_intermediate_eval(i) * vars_base.len() * SIZE_F;
        let addr_intermediate_prod =
            vars_base.addr_local_wires + wires_intermediate_prod(i) * vars_base.len() * SIZE_F;
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            addr_intermediate_eval,
            addr_computed_eval,
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;
        vec_ops.extend(VecOpExtension::<D>::mul(
            mem,
            vars_base.len(),
            addr_intermediate_prod,
            addr_computed_prod,
            addr_res_batch,
            VecOpSrc::VV,
            false,
        ));
        addr_res_batch += vars_base.len() * SIZE_F * D;

        let start_index = 1 + (degree - 1) * (i + 1);
        let end_index = (start_index + degree - 1).min(num_points());

        for j in start_index..end_index {
            vec_ops.extend(VecOpExtension::<D>::scalar_mul(
                vars_base.len(),
                values[j],
                0,
                addr_weighted_values[j - start_index],
                VecOpSrc::VV,
                false,
            ));
        }
        for j in start_index..end_index {
            vec_ops.extend(VecOpExtension::<D>::scalar_mul(
                1,
                0,
                addr_domain + j * SIZE_F,
                addr_term,
                VecOpSrc::VS,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::sub(
                vars_base.len(),
                addr_shifted_evaluation_point,
                addr_term,
                addr_term,
                VecOpSrc::VS,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::mul(
                mem,
                vars_base.len(),
                addr_weighted_values[j - start_index],
                addr_computed_prod,
                addr_eval,
                VecOpSrc::VV,
                false,
            ));
            // eval * term
            vec_ops.extend(VecOpExtension::<D>::mul(
                mem,
                vars_base.len(),
                addr_computed_eval,
                addr_term,
                addr_computed_eval,
                VecOpSrc::VV,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::add(
                vars_base.len(),
                addr_eval,
                addr_computed_eval,
                addr_computed_eval,
                VecOpSrc::VV,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::mul(
                mem,
                vars_base.len(),
                addr_computed_prod,
                addr_term,
                addr_computed_prod,
                VecOpSrc::VV,
                false,
            ));
        }
    }
    let addr_evaluation_value =
        vars_base.addr_local_wires + wires_evaluation_value() * vars_base.len() * SIZE_F;
    vec_ops.extend(VecOpExtension::<D>::sub(
        vars_base.len(),
        addr_evaluation_value,
        addr_computed_eval,
        addr_res_batch,
        VecOpSrc::VV,
        true,
    ));

    mem.free("domain");
    for i in 0..degree {
        mem.free(format!("weighted_values_{}", i).as_str());
    }
    mem.free("computed_eval");
    mem.free("computed_prod");
    mem.free("eval");
    mem.free("term");

    vec_ops
}

fn get_params(id: String, param: &str) -> usize {
    if let Some(index) = id.find(param) {
        let substring = &id[index + param.len() + 2..];
        let end = substring.find(",").unwrap_or(substring.find(" ").unwrap());
        let value = &substring[..end];
        value.parse::<usize>().unwrap()
    } else {
        panic!("Gate {:?} doesn't have parameter {:?} !", id, param);
    }
}
