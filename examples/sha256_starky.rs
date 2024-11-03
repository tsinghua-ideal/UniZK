use env_logger::Env;
use log::info;
use std::env;
use std::iter::once;

use plonky2::field::types::Field;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use unizk::config::RamConfig;
use unizk::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};
use unizk::memory::memory_allocator::MemAlloc;
use unizk::starky::constraint_consumer::ConstraintConsumer;
use unizk::starky::prover::prove;
use unizk::starky::stark::EvaluationFrame;
use unizk::system::system::System;
use unizk::util::{BATCH_SIZE, SIZE_F};
use starky::config::StarkConfig;
use starky::sha256::layout::*;
use starky::sha256::{Sha2CompressionStark, Sha2StarkCompressor};
use starky::util::to_u32_array_be;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = Sha2CompressionStark<F, D>; 

fn eval_packed_generic(
    sys: &mut System,
    vars: &EvaluationFrame,
    yield_constr: &ConstraintConsumer,
) -> Vec<VecOpConfig> {
    let mut res = Vec::new();
    let curr_row = |idx: usize| vars.addr_local_values + idx * BATCH_SIZE * SIZE_F;
    let next_row = |idx: usize| vars.addr_next_values + idx * BATCH_SIZE * SIZE_F;
    let addr_constraint = sys.mem.get_addr("constraint").unwrap();

    // set hash idx to 1 at the start. hash_idx should be 1-indexed.
    // yield_constr.constraint_first_row(P::ONES - curr_row[HASH_IDX]);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: curr_row(HASH_IDX),
        addr_input_1: 0,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    res.extend(yield_constr.constraint_first_row(addr_constraint));

    // set his to initial values at start of hash
    let is_hash_start = curr_row(step_bit(0));
    for i in 0..8 {
        // degree 3
        // yield_constr
        //     .constraint(is_hash_start * (curr_row[h_i(i)] - FE::from_canonical_u32(HASH_IV[i])));
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(h_i(i)),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: is_hash_start,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // ensure his stay the same outside last two rows of hash
    // let his_should_change = next_row[step_bit(64)] + curr_row[step_bit(64)];
    let addr_his_should_change = sys
        .mem
        .alloc("his_should_change", BATCH_SIZE * SIZE_F)
        .unwrap();
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: next_row(step_bit(64)),
        addr_input_1: curr_row(step_bit(64)),
        addr_output: addr_his_should_change,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_his_should_change,
        addr_input_1: 0,
        addr_output: addr_his_should_change,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    for i in 0..8 {
        // degree 2
        // yield_constr.constraint_transition(
        //     (P::ONES - his_should_change) * (next_row[h_i(i)] - curr_row[h_i(i)]),
        // );
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(h_i(i)),
            addr_input_1: curr_row(h_i(i)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: addr_his_should_change,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // let next_is_phase_0: P = (0..16).map(|i| next_row[step_bit(i)]).sum();
    // let next_is_phase_1: P = (16..64).map(|i| next_row[step_bit(i)]).sum();
    // let next_is_not_padding = next_is_phase_0 + next_is_phase_1 + next_row[step_bit(64)];
    let addr_next_is_phase_0 = sys
        .mem
        .alloc("next_is_phase_0", BATCH_SIZE * SIZE_F)
        .unwrap();
    for i in 0..16 {
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: if i != 0 { addr_next_is_phase_0 } else { 0 },
            addr_input_1: next_row(step_bit(i)),
            addr_output: addr_next_is_phase_0,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        });
    }

    let addr_next_is_phase_1 = sys
        .mem
        .alloc("next_is_phase_1", BATCH_SIZE * SIZE_F)
        .unwrap();
    for i in 16..64 {
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: if i != 16 { addr_next_is_phase_1 } else { 0 },
            addr_input_1: next_row(step_bit(i)),
            addr_output: addr_next_is_phase_1,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        });
    }

    let addr_next_is_not_padding = sys
        .mem
        .alloc("next_is_not_padding", BATCH_SIZE * SIZE_F)
        .unwrap();
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_is_phase_0,
        addr_input_1: addr_next_is_phase_1,
        addr_output: addr_next_is_not_padding,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_is_not_padding,
        addr_input_1: next_row(step_bit(64)),
        addr_output: addr_next_is_not_padding,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });

    // increment hash idx if we're at the last step and next isn't padding
    // let is_last_step = curr_row[step_bit(64)];
    let is_last_step = curr_row(step_bit(64));
    // let transition_to_next_hash = is_last_step * next_is_not_padding;
    let addr_transition_to_next_hash = sys
        .mem
        .alloc("transition_to_next_hash", BATCH_SIZE * SIZE_F)
        .unwrap();
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: is_last_step,
        addr_input_1: addr_next_is_not_padding,
        addr_output: addr_transition_to_next_hash,
        is_final_output: false,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
    });

    // degree 3
    // yield_constr.constraint_transition(
    //     transition_to_next_hash * (P::ONES - (next_row[HASH_IDX] - curr_row[HASH_IDX])),
    // );
    let addr_curr_next_row_hash_idx = sys
        .mem
        .alloc("curr_next_row_hash_idx", BATCH_SIZE * SIZE_F)
        .unwrap();
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: next_row(HASH_IDX),
        addr_input_1: curr_row(HASH_IDX),
        addr_output: addr_curr_next_row_hash_idx,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_curr_next_row_hash_idx,
        addr_input_1: 0,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_transition_to_next_hash,
        addr_input_1: addr_constraint,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // otherwise ensure hash idx stays the same unless next row is padding
    // degree 3
    // yield_constr.constraint_transition(
    //     (P::ONES - is_last_step) * next_is_not_padding * (next_row[HASH_IDX] - curr_row[HASH_IDX]),
    // );
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: is_last_step,
        addr_input_1: 0,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_constraint,
        addr_input_1: addr_next_is_not_padding,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_constraint,
        addr_input_1: addr_curr_next_row_hash_idx,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint_transition(addr_constraint));

    let addr_bit_decomp_32_tmp = sys
        .mem
        .alloc("bit_decomp_32_tmp", BATCH_SIZE * SIZE_F)
        .unwrap();
    let bit_decomp_32 = |vec_ops: &mut Vec<VecOpConfig>,
                         addr,
                         row: &dyn Fn(usize) -> usize,
                         clo_fn: fn(usize) -> usize| {
        for bit in 0..32 {
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: if bit != 0 { addr } else { 0 },
                addr_input_1: 0,
                addr_output: addr_bit_decomp_32_tmp,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr,
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            });
        }
    };

    // load input into wis rotated left by one at start
    // wi
    // let decomp = bit_decomp_32!(curr_row, wi_bit, FE, P)
    //     + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
    // yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(0)]));
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_bit);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(HASH_IDX),
            addr_input_1: 0,
            addr_output: addr_bit_decomp_32_tmp,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_bit_decomp_32_tmp,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(input_i(0)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_hash_start,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // wi minus 2
    // let decomp = bit_decomp_32!(curr_row, wi_minus_2_bit, FE, P)
    //     + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
    // yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(14)]));
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_minus_2_bit);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(HASH_IDX),
            addr_input_1: 0,
            addr_output: addr_bit_decomp_32_tmp,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_bit_decomp_32_tmp,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(input_i(14)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_hash_start,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // wi minus 15
    // let decomp = bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P)
    //     + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
    // yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(1)]));
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_minus_15_bit);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(HASH_IDX),
            addr_input_1: 0,
            addr_output: addr_bit_decomp_32_tmp,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_bit_decomp_32_tmp,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(input_i(1)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_hash_start,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // the rest
    for i in (1..=12).chain(once(14)) {
        // let decomp = curr_row[wi_field(i)] + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);

        // yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(i + 1)]));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(HASH_IDX),
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: curr_row(wi_field(i)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: curr_row(input_i(i + 1)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: is_hash_start,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // set input filter to 1 iff we're at the start, zero otherwise
    // yield_constr.constraint(is_hash_start * (P::ONES - curr_row[INPUT_FILTER]));
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: 0,
            addr_input_1: curr_row(INPUT_FILTER),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_hash_start,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));
    // yield_constr.constraint((P::ONES - is_hash_start) * curr_row[INPUT_FILTER]);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_hash_start,
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(INPUT_FILTER),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // rotate wis when next step is phase 0 and we're not starting a new hash
    // let rotate_wis = next_is_phase_0 * (P::ONES - next_row[step_bit(0)]);
    let addr_rotate_wis = sys.mem.alloc("rotate_wis", BATCH_SIZE * SIZE_F).unwrap();
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(step_bit(0)),
            addr_input_1: 0,
            addr_output: addr_rotate_wis,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_rotate_wis,
            addr_input_1: addr_next_is_phase_0,
            addr_output: addr_rotate_wis,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    // shift wis left when next step is phase 1
    // let shift_wis = next_is_phase_1;
    let addr_shift_wis = addr_next_is_phase_1;

    // wi next when rotating
    // let c =
    //     bit_decomp_32!(next_row, wi_bit, FE, P) - bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P);
    let addr_c = sys.mem.alloc("c", BATCH_SIZE * SIZE_F).unwrap();
    bit_decomp_32(&mut res, addr_c, &next_row, wi_bit);
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_minus_15_bit);

    // yield_constr.constraint_transition(rotate_wis * c);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_c,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: addr_rotate_wis,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // wi_minus_2 next
    // let decomp = bit_decomp_32!(next_row, wi_minus_2_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &next_row, wi_minus_2_bit);

    // yield_constr
    //     .constraint_transition((rotate_wis + shift_wis) * (decomp - curr_row[wi_field(14)]));
    let addr_rotate_wis_plus_shift_wis = sys
        .mem
        .alloc("rotate_wis_plus_shift_wis", BATCH_SIZE * SIZE_F)
        .unwrap();
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(wi_field(14)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_shift_wis,
            addr_input_1: addr_rotate_wis,
            addr_output: addr_rotate_wis_plus_shift_wis,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_rotate_wis_plus_shift_wis,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // wi_minus_15 next
    // let decomp = bit_decomp_32!(next_row, wi_minus_15_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &next_row, wi_minus_15_bit);
    // yield_constr.constraint_transition((rotate_wis + shift_wis) * (decomp - curr_row[wi_field(1)]));
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(wi_field(1)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_rotate_wis_plus_shift_wis,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // wi_minus_1 next
    // let decomp = bit_decomp_32!(curr_row, wi_bit, FE, P);
    // yield_constr
    //     .constraint_transition((rotate_wis + shift_wis) * (next_row[wi_field(14)] - decomp));
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_bit);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(wi_field(14)),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_rotate_wis_plus_shift_wis,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // wi_i_minus_3 next
    // let decomp = bit_decomp_32!(curr_row, wi_minus_2_bit, FE, P);
    // yield_constr
    //     .constraint_transition((rotate_wis + shift_wis) * (next_row[wi_field(12)] - decomp));
    bit_decomp_32(&mut res, addr_constraint, &curr_row, wi_minus_2_bit);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(wi_field(12)),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_rotate_wis_plus_shift_wis,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    for i in 1..12 {
        // yield_constr.constraint_transition(
        //     (rotate_wis + shift_wis) * (next_row[wi_field(i)] - curr_row[wi_field(i + 1)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(wi_field(i)),
                addr_input_1: curr_row(wi_field(i + 1)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_rotate_wis_plus_shift_wis,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // round fn in phase 0 or 1
    // let is_phase_0_or_1: P = (0..64).map(|i| curr_row[step_bit(i)]).sum();
    let addr_is_phase_0_or_1 = sys
        .mem
        .alloc("is_phase_0_or_1", BATCH_SIZE * SIZE_F)
        .unwrap();
    for i in 0..64 {
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: if i != 0 { addr_is_phase_0_or_1 } else { 0 },
            addr_input_1: curr_row(step_bit(i)),
            addr_output: addr_is_phase_0_or_1,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        });
    }

    let xor_gen = |vec_ops: &mut Vec<VecOpConfig>, addr_x: usize, addr_y: usize| {
        vec_ops.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_x,
                addr_input_1: addr_y,
                addr_output: addr_bit_decomp_32_tmp,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_y,
                addr_input_1: 0,
                addr_output: addr_c,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_c,
                addr_input_1: addr_x,
                addr_output: addr_c,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_bit_decomp_32_tmp,
                addr_input_1: addr_c,
                addr_output: addr_bit_decomp_32_tmp,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
    };

    // S1 := (e >>> 6) xor (e >>> 11) xor (e >>> 25)
    for bit in 0..32 {
        // let computed_bit = xor_gen(
        //     curr_row[e_bit((bit + 6) % 32)],
        //     curr_row[e_bit((bit + 11) % 32)],
        // );
        xor_gen(
            &mut res,
            curr_row(e_bit((bit + 6) % 32)),
            curr_row(e_bit((bit + 11) % 32)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[xor_tmp_2_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_2_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = xor_gen(
        //     curr_row[xor_tmp_2_bit(bit)],
        //     curr_row[e_bit((bit + 25) % 32)],
        // );
        xor_gen(
            &mut res,
            curr_row(xor_tmp_2_bit(bit)),
            curr_row(e_bit((bit + 25) % 32)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[big_s1_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(big_s1_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // ch := (e and f) xor ((not e) and g)
    for bit in 0..32 {
        // let computed_bit = curr_row[e_bit(bit)] * curr_row[f_bit(bit)];
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(e_bit(bit)),
            addr_input_1: curr_row(f_bit(bit)),
            addr_output: addr_bit_decomp_32_tmp,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[e_and_f_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(e_and_f_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = (P::ONES - curr_row[e_bit(bit)]) * curr_row[g_bit(bit)];
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(e_bit(bit)),
                addr_input_1: 0,
                addr_output: addr_bit_decomp_32_tmp,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_bit_decomp_32_tmp,
                addr_input_1: curr_row(g_bit(bit)),
                addr_output: addr_bit_decomp_32_tmp,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[not_e_and_g_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(not_e_and_g_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = xor_gen(curr_row[e_and_f_bit(bit)], curr_row[not_e_and_g_bit(bit)]);
        xor_gen(
            &mut res,
            curr_row(e_and_f_bit(bit)),
            curr_row(not_e_and_g_bit(bit)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[ch_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(ch_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // S0 := (a >>> 2) xor (a >>> 13) xor (a >>> 22)
    for bit in 0..32 {
        // let computed_bit = xor_gen(
        //     curr_row[a_bit((bit + 2) % 32)],
        //     curr_row[a_bit((bit + 13) % 32)],
        // );
        xor_gen(
            &mut res,
            curr_row(a_bit((bit + 2) % 32)),
            curr_row(a_bit((bit + 13) % 32)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[xor_tmp_3_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_3_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = xor_gen(
        //     curr_row[xor_tmp_3_bit(bit)],
        //     curr_row[a_bit((bit + 22) % 32)],
        // );
        xor_gen(
            &mut res,
            curr_row(xor_tmp_3_bit(bit)),
            curr_row(a_bit((bit + 22) % 32)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[big_s0_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(big_s0_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // maj := (a and b) xor (a and c) xor (b and c)
    for bit in 0..32 {
        // degree 3
        // yield_constr.constraint(
        //     is_phase_0_or_1
        //         * (curr_row[a_and_b_bit(bit)] - curr_row[a_bit(bit)] * curr_row[b_bit(bit)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_bit(bit)),
                addr_input_1: curr_row(b_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_and_b_bit(bit)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // degree 3
        // yield_constr.constraint(
        //     is_phase_0_or_1
        //         * (curr_row[a_and_c_bit(bit)] - curr_row[a_bit(bit)] * curr_row[c_bit(bit)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_bit(bit)),
                addr_input_1: curr_row(c_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_and_c_bit(bit)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // degree 3
        // yield_constr.constraint(
        //     is_phase_0_or_1
        //         * (curr_row[b_and_c_bit(bit)] - curr_row[b_bit(bit)] * curr_row[c_bit(bit)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(b_bit(bit)),
                addr_input_1: curr_row(c_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(b_and_c_bit(bit)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = xor_gen(curr_row[a_and_b_bit(bit)], curr_row[a_and_c_bit(bit)]);
        xor_gen(
            &mut res,
            curr_row(a_and_b_bit(bit)),
            curr_row(a_and_c_bit(bit)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[xor_tmp_4_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_4_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // let computed_bit = xor_gen(curr_row[xor_tmp_4_bit(bit)], curr_row[b_and_c_bit(bit)]);
        xor_gen(
            &mut res,
            curr_row(xor_tmp_4_bit(bit)),
            curr_row(b_and_c_bit(bit)),
        );
        // degree 3
        // yield_constr.constraint(is_phase_0_or_1 * (curr_row[maj_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(maj_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // set round constant
    for step in 0..64 {
        // degree 2
        // yield_constr.constraint(
        //     curr_row[step_bit(step)]
        //         * (curr_row[KI] - FE::from_canonical_u32(ROUND_CONSTANTS[step])),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(KI),
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: curr_row(step_bit(step)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // temp1 := h + S1 + ch + k[i] + w[i]
    let addr_temp1_minus_ki = sys
        .mem
        .alloc("temp1_minus_ki", BATCH_SIZE * SIZE_F)
        .unwrap();
    // e := d + temp1
    let h_field = curr_row(H_COL);
    // let big_s1_field = bit_decomp_32!(curr_row, big_s1_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &curr_row, big_s1_bit);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: h_field,
        addr_input_1: addr_c,
        addr_output: addr_temp1_minus_ki,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });

    // let ch_field = bit_decomp_32!(curr_row, ch_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &curr_row, ch_bit);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_temp1_minus_ki,
        addr_input_1: addr_c,
        addr_output: addr_temp1_minus_ki,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });

    // let wi_u32 = bit_decomp_32!(curr_row, wi_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &curr_row, wi_bit);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_temp1_minus_ki,
        addr_input_1: addr_c,
        addr_output: addr_temp1_minus_ki,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });

    // let temp1_minus_ki = h_field + big_s1_field + ch_field + wi_u32;

    // let d_field = curr_row[D_COL];
    let d_field = curr_row(D_COL);

    // let e_u32_next = bit_decomp_32!(next_row, e_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &next_row, e_bit);

    // degree 2
    // yield_constr.constraint(
    //     is_phase_0_or_1 * (curr_row[E_NEXT_FIELD] - (d_field + temp1_minus_ki + curr_row[KI])),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: d_field,
            addr_input_1: addr_temp1_minus_ki,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(KI),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(E_NEXT_FIELD),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));
    // degree 3
    // yield_constr.constraint_transition(
    //     is_phase_0_or_1
    //         * (curr_row[E_NEXT_FIELD]
    //             - (e_u32_next + curr_row[E_NEXT_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(E_NEXT_QUOTIENT),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_c,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(E_NEXT_FIELD),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // temp2 := S0 + maj
    // a := temp1 + temp2
    // let s0_field = bit_decomp_32!(curr_row, big_s0_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &curr_row, big_s0_bit);
    // let maj_field = bit_decomp_32!(curr_row, maj_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &curr_row, maj_bit);
    // let temp2 = s0_field + maj_field;

    // degree 2
    // yield_constr.constraint(
    //     is_phase_0_or_1 * (curr_row[A_NEXT_FIELD] - (temp2 + temp1_minus_ki + curr_row[KI])),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_c,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: addr_temp1_minus_ki,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(KI),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(A_NEXT_FIELD),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // degree 3
    // let a_u32_next = bit_decomp_32!(next_row, a_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &next_row, a_bit);
    // yield_constr.constraint_transition(
    //     is_phase_0_or_1
    //         * (curr_row[A_NEXT_FIELD]
    //             - (a_u32_next + curr_row[A_NEXT_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(A_NEXT_QUOTIENT),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_c,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(A_NEXT_FIELD),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // update local vars when not in last step
    // h := g
    // g := f
    // f := e
    // d := c
    // c := b
    // b := a
    // let decomp = bit_decomp_32!(curr_row, g_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &curr_row, g_bit);
    // yield_constr.constraint_transition(is_phase_0_or_1 * (next_row[H_COL] - decomp));
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: next_row(H_COL),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // let decomp = bit_decomp_32!(curr_row, c_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &curr_row, c_bit);
    // yield_constr.constraint_transition(is_phase_0_or_1 * (next_row[D_COL] - decomp));
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: next_row(D_COL),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_phase_0_or_1,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    for bit in 0..32 {
        // degree 2
        // yield_constr
        //     .constraint_transition(is_phase_0_or_1 * (next_row[g_bit(bit)] - curr_row[f_bit(bit)]));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(g_bit(bit)),
                addr_input_1: curr_row(f_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
        // yield_constr
        //     .constraint_transition(is_phase_0_or_1 * (next_row[f_bit(bit)] - curr_row[e_bit(bit)]));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(f_bit(bit)),
                addr_input_1: curr_row(e_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
        // yield_constr
        //     .constraint_transition(is_phase_0_or_1 * (next_row[c_bit(bit)] - curr_row[b_bit(bit)]));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(c_bit(bit)),
                addr_input_1: curr_row(b_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
        // yield_constr
        //     .constraint_transition(is_phase_0_or_1 * (next_row[b_bit(bit)] - curr_row[a_bit(bit)]));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(b_bit(bit)),
                addr_input_1: curr_row(a_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_phase_0_or_1,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // update his in last step of phase 1
    let update_his = curr_row(step_bit(63));
    // let vars = [
    //     bit_decomp_32!(next_row, a_bit, FE, P),
    //     bit_decomp_32!(next_row, b_bit, FE, P),
    //     bit_decomp_32!(next_row, c_bit, FE, P),
    //     next_row[D_COL],
    //     bit_decomp_32!(next_row, e_bit, FE, P),
    //     bit_decomp_32!(next_row, f_bit, FE, P),
    //     bit_decomp_32!(next_row, g_bit, FE, P),
    //     next_row[H_COL],
    // ];
    let vars_0 = vec![a_bit, b_bit, c_bit];
    let vars_1 = vec![e_bit, f_bit, g_bit];

    for i in 0..8 {
        // degree 2
        let addr_var = if i < 3 {
            bit_decomp_32(&mut res, addr_constraint, &next_row, vars_0[i]);
            addr_constraint
        } else if i == 3 {
            next_row(D_COL)
        } else if i < 7 {
            bit_decomp_32(&mut res, addr_constraint, &next_row, vars_1[i - 4]);
            addr_constraint
        } else {
            next_row(H_COL)
        };

        // yield_constr.constraint_transition(
        //     update_his * (curr_row[h_i_next_field(i)] - (curr_row[h_i(i)] + vars[i])),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(h_i(i)),
                addr_input_1: addr_var,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(h_i_next_quotient(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: update_his,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));

        // degree 3
        // yield_constr.constraint_transition(
        //     update_his
        //         * (curr_row[h_i_next_field(i)]
        //             - (next_row[h_i(i)]
        //                 + curr_row[h_i_next_quotient(i)] * FE::from_canonical_u64(1 << 32))),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(h_i_next_quotient(i)),
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(h_i(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(h_i_next_field(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: update_his,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // set output to his during last step, 0 otherwise
    for i in 0..8 {
        // degree 2
        // yield_constr.constraint(
        //     is_last_step
        //         * (curr_row[output_i(i)]
        //             - (curr_row[h_i(i)] + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32))),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(HASH_IDX),
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(h_i(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(output_i(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: is_last_step,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
        // yield_constr.constraint((P::ONES - is_last_step) * curr_row[output_i(i)])
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: is_last_step,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(output_i(i)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // set output filter to 1 iff it's the last step, zero otherwise
    // yield_constr.constraint(is_last_step * (P::ONES - curr_row[OUTPUT_FILTER]));
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: 0,
            addr_input_1: curr_row(OUTPUT_FILTER),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_last_step,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));
    // yield_constr.constraint((P::ONES - is_last_step) * curr_row[OUTPUT_FILTER]);
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: is_last_step,
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(OUTPUT_FILTER),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // message schedule to get next row's wi when next row is in phase 1

    let addr_do_msg_schedule = addr_next_is_phase_1;

    // s0 := (w[i-15] >>> 7) xor (w[i-15] >>> 18) xor (w[i-15] >> 3)
    for bit in 0..29 {
        // let computed_bit = xor_gen(
        //     next_row[wi_minus_15_bit((bit + 7) % 32)],
        //     next_row[wi_minus_15_bit((bit + 18) % 32)],
        // );
        xor_gen(
            &mut res,
            next_row(wi_minus_15_bit((bit + 7) % 32)),
            next_row(wi_minus_15_bit((bit + 18) % 32)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[xor_tmp_0_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(xor_tmp_0_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));

        // let computed_bit = xor_gen(
        //     next_row[xor_tmp_0_bit(bit)],
        //     next_row[wi_minus_15_bit(bit + 3)],
        // );
        xor_gen(
            &mut res,
            next_row(xor_tmp_0_bit(bit)),
            next_row(wi_minus_15_bit(bit + 3)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[little_s0_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(little_s0_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }
    for bit in 29..32 {
        // we can ignore the second XOR in this case since it's with 0
        // let computed_bit = xor_gen(
        //     next_row[wi_minus_15_bit((bit + 7) % 32)],
        //     next_row[wi_minus_15_bit((bit + 18) % 32)],
        // );
        xor_gen(
            &mut res,
            next_row(wi_minus_15_bit((bit + 7) % 32)),
            next_row(wi_minus_15_bit((bit + 18) % 32)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[little_s0_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(little_s0_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // s1 := (w[i-2] >>> 17) xor (w[i-2] >>> 19) xor (w[i-2] >> 10)
    for bit in 0..22 {
        // let computed_bit = xor_gen(
        //     next_row[wi_minus_2_bit((bit + 17) % 32)],
        //     next_row[wi_minus_2_bit((bit + 19) % 32)],
        // );
        xor_gen(
            &mut res,
            next_row(wi_minus_2_bit((bit + 17) % 32)),
            next_row(wi_minus_2_bit((bit + 19) % 32)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[xor_tmp_1_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(xor_tmp_1_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));

        // let computed_bit = xor_gen(
        //     next_row[xor_tmp_1_bit(bit)],
        //     next_row[wi_minus_2_bit(bit + 10)],
        // );
        xor_gen(
            &mut res,
            next_row(xor_tmp_1_bit(bit)),
            next_row(wi_minus_2_bit(bit + 10)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[little_s1_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(little_s1_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }
    for bit in 22..32 {
        // we can ignore the second XOR in this case since it's with 0
        // let computed_bit = xor_gen(
        //     next_row[wi_minus_2_bit((bit + 17) % 32)],
        //     next_row[wi_minus_2_bit((bit + 19) % 32)],
        // );
        xor_gen(
            &mut res,
            next_row(wi_minus_2_bit((bit + 17) % 32)),
            next_row(wi_minus_2_bit((bit + 19) % 32)),
        );
        // degree 3
        // yield_constr
        //     .constraint_transition(do_msg_schedule * (next_row[little_s1_bit(bit)] - computed_bit));
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(little_s1_bit(bit)),
                addr_input_1: addr_bit_decomp_32_tmp,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_do_msg_schedule,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // w[i] := w[i-16] + s0 + w[i-7] + s1

    // degree 1
    // let s0_field_computed = bit_decomp_32!(next_row, little_s0_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &next_row, little_s0_bit);
    // let s1_field_computed = bit_decomp_32!(next_row, little_s1_bit, FE, P);
    bit_decomp_32(&mut res, addr_constraint, &next_row, little_s1_bit);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_c,
        addr_input_1: addr_constraint,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });
    // let wi_minus_16_field = bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &curr_row, wi_minus_15_bit);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_c,
        addr_input_1: addr_constraint,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });
    // let wi_minus_7_field = next_row[wi_field(8)];
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: next_row(wi_field(8)),
        addr_input_1: addr_constraint,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::ADD,
        op_src: VecOpSrc::VV,
    });

    // degree 2
    // yield_constr.constraint_transition(
    //     do_msg_schedule
    //         * (next_row[WI_FIELD]
    //             - (wi_minus_16_field + s0_field_computed + wi_minus_7_field + s1_field_computed)),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: next_row(WI_FIELD),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_do_msg_schedule,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint_transition(addr_constraint));

    // let wi = bit_decomp_32!(next_row, wi_bit, FE, P);
    bit_decomp_32(&mut res, addr_c, &next_row, wi_bit);
    // degree 3
    // yield_constr.constraint(
    //     do_msg_schedule
    //         * (next_row[WI_FIELD] - (wi + next_row[WI_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
    // );
    res.extend(vec![
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(WI_QUOTIENT),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_c,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(WI_FIELD),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        },
        VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_do_msg_schedule,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        },
    ]);
    res.extend(yield_constr.constraint(addr_constraint));

    // set initial step bits to a 1 followed by NUM_STEPS_PER_HASH-1 0s
    // yield_constr.constraint_first_row(P::ONES - curr_row[step_bit(0)]);
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: curr_row(step_bit(0)),
        addr_input_1: 0,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint_first_row(addr_constraint));
    for step in 1..NUM_STEPS_PER_HASH {
        // yield_constr.constraint_first_row(curr_row[step_bit(step)]);
        res.extend(yield_constr.constraint_first_row(curr_row(step_bit(step))));
    }

    // inc step bits when next is not padding
    for bit in 0..NUM_STEPS_PER_HASH {
        // degree 3
        // yield_constr.constraint_transition(
        //     next_is_not_padding
        //         * (next_row[step_bit((bit + 1) % NUM_STEPS_PER_HASH)] - curr_row[step_bit(bit)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(step_bit((bit + 1) % NUM_STEPS_PER_HASH)),
                addr_input_1: curr_row(step_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_next_is_not_padding,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }

    // step bits
    for bit in 0..NUM_STEPS_PER_HASH {
        // yield_constr.constraint((P::ONES - curr_row[step_bit(bit)]) * curr_row[step_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(step_bit(bit)),
                addr_input_1: curr_row(step_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    for bit in 0..32 {
        // wi
        // yield_constr.constraint((P::ONES - curr_row[wi_bit(bit)]) * curr_row[wi_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(wi_bit(bit)),
                addr_input_1: curr_row(wi_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // wi minus 2
        // yield_constr
        //     .constraint((P::ONES - curr_row[wi_minus_2_bit(bit)]) * curr_row[wi_minus_2_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(wi_minus_2_bit(bit)),
                addr_input_1: curr_row(wi_minus_2_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // wi minus 15
        // yield_constr.constraint(
        //     (P::ONES - curr_row[wi_minus_15_bit(bit)]) * curr_row[wi_minus_15_bit(bit)],
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(wi_minus_15_bit(bit)),
                addr_input_1: curr_row(wi_minus_15_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // s0
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[little_s0_bit(bit)]) * curr_row[little_s0_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(little_s0_bit(bit)),
                addr_input_1: curr_row(little_s0_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // s1
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[little_s1_bit(bit)]) * curr_row[little_s1_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(little_s1_bit(bit)),
                addr_input_1: curr_row(little_s1_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // a
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[a_bit(bit)]) * curr_row[a_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_bit(bit)),
                addr_input_1: curr_row(a_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // b
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[b_bit(bit)]) * curr_row[b_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(b_bit(bit)),
                addr_input_1: curr_row(b_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // c
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[c_bit(bit)]) * curr_row[c_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(c_bit(bit)),
                addr_input_1: curr_row(c_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // e
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[e_bit(bit)]) * curr_row[e_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(e_bit(bit)),
                addr_input_1: curr_row(e_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // f
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[f_bit(bit)]) * curr_row[f_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(f_bit(bit)),
                addr_input_1: curr_row(f_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // g
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[g_bit(bit)]) * curr_row[g_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(g_bit(bit)),
                addr_input_1: curr_row(g_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // S0
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[big_s0_bit(bit)]) * curr_row[big_s0_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(big_s0_bit(bit)),
                addr_input_1: curr_row(big_s0_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // S1
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[big_s1_bit(bit)]) * curr_row[big_s1_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(big_s1_bit(bit)),
                addr_input_1: curr_row(big_s1_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // (not e) and g
    for bit in 0..32 {
        // yield_constr.constraint(
        //     (P::ONES - curr_row[not_e_and_g_bit(bit)]) * curr_row[not_e_and_g_bit(bit)],
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(not_e_and_g_bit(bit)),
                addr_input_1: curr_row(not_e_and_g_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // e and f
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[e_and_f_bit(bit)]) * curr_row[e_and_f_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(e_and_f_bit(bit)),
                addr_input_1: curr_row(e_and_f_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // ch
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[ch_bit(bit)]) * curr_row[ch_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(ch_bit(bit)),
                addr_input_1: curr_row(ch_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // a and b
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[a_and_b_bit(bit)]) * curr_row[a_and_b_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_and_b_bit(bit)),
                addr_input_1: curr_row(a_and_b_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // a and c
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[a_and_c_bit(bit)]) * curr_row[a_and_c_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(a_and_c_bit(bit)),
                addr_input_1: curr_row(a_and_c_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // b and c
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[b_and_c_bit(bit)]) * curr_row[b_and_c_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(b_and_c_bit(bit)),
                addr_input_1: curr_row(b_and_c_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // maj
    for bit in 0..32 {
        // yield_constr.constraint((P::ONES - curr_row[maj_bit(bit)]) * curr_row[maj_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(maj_bit(bit)),
                addr_input_1: curr_row(maj_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // tmps
    for bit in 0..29 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[xor_tmp_0_bit(bit)]) * curr_row[xor_tmp_0_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_0_bit(bit)),
                addr_input_1: curr_row(xor_tmp_0_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }
    for bit in 0..22 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[xor_tmp_1_bit(bit)]) * curr_row[xor_tmp_1_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_1_bit(bit)),
                addr_input_1: curr_row(xor_tmp_1_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }
    for bit in 0..32 {
        // yield_constr
        //     .constraint((P::ONES - curr_row[xor_tmp_2_bit(bit)]) * curr_row[xor_tmp_2_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_2_bit(bit)),
                addr_input_1: curr_row(xor_tmp_2_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // yield_constr
        //     .constraint((P::ONES - curr_row[xor_tmp_3_bit(bit)]) * curr_row[xor_tmp_3_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_3_bit(bit)),
                addr_input_1: curr_row(xor_tmp_3_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));

        // yield_constr
        //     .constraint((P::ONES - curr_row[xor_tmp_4_bit(bit)]) * curr_row[xor_tmp_4_bit(bit)]);
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(xor_tmp_4_bit(bit)),
                addr_input_1: curr_row(xor_tmp_4_bit(bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
        res.extend(yield_constr.constraint(addr_constraint));
    }

    sys.mem.free("his_should_change");
    sys.mem.free("next_is_phase_0");
    sys.mem.free("next_is_phase_1");
    sys.mem.free("next_is_not_padding");
    sys.mem.free("transition_to_next_hash");
    sys.mem.free("curr_next_row_hash_idx");
    sys.mem.free("bit_decomp_32_tmp");
    sys.mem.free("rotate_wis");
    sys.mem.free("rotate_wis_plus_shift_wis");
    sys.mem.free("is_phase_0_or_1");
    sys.mem.free("temp1_minus_ki");
    sys.mem.free("c");
    res
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let args: Vec<String> = env::args().collect();
    let num_hashes = args[1].parse::<i32>().unwrap();
    println!(
        "\n============== num hashes {} =======================================",
        num_hashes
    );

    let mut ramsim = RamConfig::new(&format!("{}", "sha256_starky"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(32, 4096);
    let mut sys = System::new(mem, ramsim);

    let mut compressor = Sha2StarkCompressor::new();
    let _zero_bytes = [0; 32];
    let init_left = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x80,
    ];
    let init_right = [
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0xf8,
    ];
    for i in 0..num_hashes {
        let mut left = to_u32_array_be::<8>(init_left);
        let right = to_u32_array_be::<8>(init_right);
        left[0] = i as u32;

        compressor.add_instance(left, right);
    }
    let trace = compressor.generate();

    let config = StarkConfig::standard_fast_config();
    let stark = S::new();
    prove::<F, C, S, D>(&mut sys, stark, &config, trace, &[], eval_packed_generic);

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}
