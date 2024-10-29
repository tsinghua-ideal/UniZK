use env_logger::Env;
use log::info;

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
use starky::aes128::constants::*;
use starky::aes128::generation::AesTraceGenerator;
use starky::aes128::layout::*;
use starky::aes128::AesStark;
use starky::aes128::{key_3_after_rot_start, key_u8_bit};
use starky::config::StarkConfig;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = AesStark<F, D>;

fn eval_packed_generic(
    sys: &mut System,
    vars: &EvaluationFrame,
    yield_constr: &ConstraintConsumer,
) -> Vec<VecOpConfig> {
    let mut res = Vec::new();
    let curr_row = |idx: usize| vars.addr_local_values + idx * BATCH_SIZE * SIZE_F;
    let next_row = |idx: usize| vars.addr_next_values + idx * BATCH_SIZE * SIZE_F;
    let addr_constraint = sys.mem.get_addr("constraint").unwrap();
    let addr_xor_gen_tmp = sys.mem.alloc("xor_gen_tmp", BATCH_SIZE * SIZE_F).unwrap();
    let addr_xor_gen = sys.mem.alloc("xor_gen", BATCH_SIZE * SIZE_F).unwrap();

    let addr_a_j_b_i = sys.mem.alloc("a_j_b_i", BATCH_SIZE * SIZE_F).unwrap();
    let addr_not_hi_bit_set = sys
        .mem
        .alloc("not_hi_bit_set", BATCH_SIZE * SIZE_F)
        .unwrap();

    let addr_is_gmul_substep = sys
        .mem
        .alloc("is_gmul_substep", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_sub = sys.mem.alloc("is_sub", BATCH_SIZE * SIZE_F).unwrap();
    let addr_next_is_not_padding = sys
        .mem
        .alloc("next_is_not_padding", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_add_round_key = sys
        .mem
        .alloc("is_add_round_key", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_key_xor_rcon = sys
        .mem
        .alloc("is_key_xor_rcon", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_not_ket_xor_rcon = sys
        .mem
        .alloc("is_not_ket_xor_rcon", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_state_shift_rows = sys
        .mem
        .alloc("is_state_shift_rows", BATCH_SIZE * SIZE_F)
        .unwrap();
    let addr_is_state_mix_columns = sys
        .mem
        .alloc("is_state_mix_columns", BATCH_SIZE * SIZE_F)
        .unwrap();

    let xor_gen = |vec_ops: &mut Vec<VecOpConfig>, addr_x: usize, addr_y: usize| {
        vec_ops.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_x,
                addr_input_1: addr_y,
                addr_output: addr_xor_gen,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_y,
                addr_input_1: 0,
                addr_output: addr_xor_gen_tmp,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen_tmp,
                addr_input_1: addr_x,
                addr_output: addr_xor_gen_tmp,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            },
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: addr_xor_gen_tmp,
                addr_output: addr_xor_gen,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            },
        ]);
    };

    let check_gmul = |vec_ops: &mut Vec<VecOpConfig>,
                      sub_step: usize,
                      a_start: usize,
                      b_start: usize,
                      tmp_start: usize| {
        assert!(sub_step < 7);

        // a * b_i
        if sub_step == 0 {
            let addr_b_i = curr_row(b_start);
            for j in 0..8 {
                let addr_a_j = curr_row(a_start + j);
                // let a_j_b_i = a_j * b_i;

                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_a_j,
                    addr_input_1: addr_b_i,
                    addr_output: addr_a_j_b_i,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });

                // yield_constr.constraint(
                //     is_gmul_substep * (curr_row[key_gmul_a_bi_bit(tmp_start, j)] - a_j_b_i),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(key_gmul_a_bi_bit(tmp_start, j)),
                    addr_input_1: addr_a_j_b_i,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }
        }
        let addr_b_i = curr_row(b_start + sub_step + 1);
        for j in 0..8 {
            let addr_a_j = curr_row(key_gmul_a_tmp_bit(tmp_start, j));
            // let a_j_b_i = a_j * b_i;
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_a_j,
                addr_input_1: addr_b_i,
                addr_output: addr_a_j_b_i,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            // yield_constr.constraint_transition(
            //     is_gmul_substep * (next_row[key_gmul_a_bi_bit(tmp_start, j)] - a_j_b_i),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(key_gmul_a_bi_bit(tmp_start, j)),
                addr_input_1: addr_a_j_b_i,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }

        // p ^= a
        for j in 0..8 {
            if sub_step == 0 {
                let addr_a_j = curr_row(key_gmul_a_bi_bit(tmp_start, j));
                // yield_constr
                //     .constraint(is_gmul_substep * (a_j - curr_row[key_gmul_p_bit(tmp_start, j)]));
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_a_j,
                    addr_input_1: curr_row(key_gmul_p_bit(tmp_start, j)),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }

            let addr_a_j = next_row(key_gmul_a_bi_bit(tmp_start, j));
            // let computed_bit = xor_gen(a_j, curr_row[key_gmul_p_bit(tmp_start, j)]);
            xor_gen(vec_ops, addr_a_j, curr_row(key_gmul_p_bit(tmp_start, j)));

            // yield_constr.constraint_transition(
            //     is_gmul_substep * (computed_bit - next_row[key_gmul_p_bit(tmp_start, j)]),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(key_gmul_p_bit(tmp_start, j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }

        // a shift xor 0x1b
        if sub_step == 0 {
            for j in 1..8 {
                let addr_a_j = curr_row(a_start + j - 1);

                xor_gen(vec_ops, addr_a_j, 0);

                // yield_constr.constraint(
                //     is_gmul_substep
                //         * (computed_bit - curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_xor_gen,
                    addr_input_1: curr_row(key_gmul_a_shift_xor_1b_bit(tmp_start, j)),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }

            // yield_constr.constraint(
            //     is_gmul_substep
            //         * (curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)]
            //             - FE::from_canonical_u32(1)),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(key_gmul_a_shift_xor_1b_bit(tmp_start, 0)),
                addr_input_1: 0,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VS,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint(addr_constraint));
        }
        for j in 1..8 {
            // let a_j = curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)];
            let addr_a_j = curr_row(key_gmul_a_tmp_bit(tmp_start, j - 1));
            // let c_bit = FE::from_canonical_u32((c >> j) & 1);
            // let computed_bit = a_j + c_bit - a_j * c_bit.doubles();
            xor_gen(vec_ops, addr_a_j, 0);
            // yield_constr.constraint_transition(
            //     is_gmul_substep
            //         * (computed_bit - next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(key_gmul_a_shift_xor_1b_bit(tmp_start, j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }
        // yield_constr.constraint_transition(
        //     is_gmul_substep
        //         * (next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)] - FE::from_canonical_u32(1)),
        // );
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(key_gmul_a_shift_xor_1b_bit(tmp_start, 0)),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
        });
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: addr_is_gmul_substep,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        vec_ops.extend(yield_constr.constraint_transition(addr_constraint));

        // a xor
        if sub_step == 0 {
            // let hi_bit_set = curr_row[a_start + 7];
            let addr_hi_bit_set = curr_row(a_start + 7);
            for j in 0..8 {
                // yield_constr.constraint(
                //     is_gmul_substep
                //         * hi_bit_set
                //         * (curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]
                //             - curr_row[key_gmul_a_tmp_bit(tmp_start, j)]),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(key_gmul_a_shift_xor_1b_bit(tmp_start, j)),
                    addr_input_1: curr_row(key_gmul_a_tmp_bit(tmp_start, j)),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_hi_bit_set,
                    addr_input_1: addr_constraint,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }
            for j in 1..8 {
                // yield_constr.constraint(
                //     is_gmul_substep
                //         * (P::ONES - hi_bit_set)
                //         * (curr_row[a_start + j - 1] - curr_row[key_gmul_a_tmp_bit(tmp_start, j)]),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(a_start + j - 1),
                    addr_input_1: curr_row(key_gmul_a_tmp_bit(tmp_start, j)),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_hi_bit_set,
                    addr_input_1: 0,
                    addr_output: addr_not_hi_bit_set,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VS,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_not_hi_bit_set,
                    addr_input_1: addr_constraint,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }
            // yield_constr.constraint(
            //     is_gmul_substep
            //         * (P::ONES - hi_bit_set)
            //         * curr_row[key_gmul_a_tmp_bit(tmp_start, 0)],
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_gmul_substep,
                addr_input_1: addr_not_hi_bit_set,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(key_gmul_a_tmp_bit(tmp_start, 0)),
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint(addr_constraint));
        }
        // let hi_bit_set = curr_row[key_gmul_a_tmp_bit(tmp_start, 7)];
        let addr_hi_bit_set = curr_row(key_gmul_a_tmp_bit(tmp_start, 7));
        for j in 0..8 {
            // yield_constr.constraint_transition(
            //     is_gmul_substep
            //         * hi_bit_set
            //         * (next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]
            //             - next_row[key_gmul_a_tmp_bit(tmp_start, j)]),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(key_gmul_a_shift_xor_1b_bit(tmp_start, j)),
                addr_input_1: next_row(key_gmul_a_tmp_bit(tmp_start, j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_hi_bit_set,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }
        for j in 1..8 {
            // yield_constr.constraint_transition(
            //     is_gmul_substep
            //         * (P::ONES - hi_bit_set)
            //         * (curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)]
            //             - next_row[key_gmul_a_tmp_bit(tmp_start, j)]),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(key_gmul_a_tmp_bit(tmp_start, j - 1)),
                addr_input_1: next_row(key_gmul_a_tmp_bit(tmp_start, j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_hi_bit_set,
                addr_input_1: 0,
                addr_output: addr_not_hi_bit_set,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_not_hi_bit_set,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }
        // yield_constr.constraint_transition(
        //     is_gmul_substep * (P::ONES - hi_bit_set) * next_row[key_gmul_a_tmp_bit(tmp_start, 0)],
        // );
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_hi_bit_set,
            addr_input_1: 0,
            addr_output: addr_not_hi_bit_set,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_gmul_substep,
            addr_input_1: addr_not_hi_bit_set,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: next_row(key_gmul_a_tmp_bit(tmp_start, 0)),
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
    };

    let check_gmul_const = |vec_ops: &mut Vec<VecOpConfig>,
                            sub_step: usize,
                            a: u8,
                            b_start: usize,
                            tmp_start: usize| {
        assert!(sub_step < 7);

        // a * b_i
        if sub_step == 0 {
            // let b_i = curr_row[b_start];
            let addr_b_i = curr_row(b_start);
            for j in 0..8 {
                // let a_j_b_i = b_i * a_j;
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_b_i,
                    addr_input_1: 0,
                    addr_output: addr_a_j_b_i,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VS,
                });
                // yield_constr.constraint(
                //     is_gmul_substep
                //         * (curr_row[state_gmul_const_a_bi_bit(tmp_start, j)] - a_j_b_i),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(state_gmul_const_a_bi_bit(tmp_start, j)),
                    addr_input_1: addr_a_j_b_i,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }
        }
        // let b_i = curr_row[b_start + sub_step + 1];
        let addr_b_i = curr_row(b_start + sub_step + 1);
        for j in 0..8 {
            // let a_j = FE::from_canonical_u32(((temp_a[sub_step + 1] >> j) & 1) as u32);
            // let a_j_b_i = b_i * a_j;
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_b_i,
                addr_input_1: 0,
                addr_output: addr_a_j_b_i,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
            });
            // yield_constr.constraint_transition(
            //     is_gmul_substep * (next_row[state_gmul_const_a_bi_bit(tmp_start, j)] - a_j_b_i),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(state_gmul_const_a_bi_bit(tmp_start, j)),
                addr_input_1: addr_a_j_b_i,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }

        // p ^= a
        for j in 0..8 {
            if sub_step == 0 {
                // let a_j = curr_row[state_gmul_const_a_bi_bit(tmp_start, j)];
                let addr_a_j = curr_row(state_gmul_const_a_bi_bit(tmp_start, j));
                // yield_constr.constraint(
                //     is_gmul_substep * (curr_row[state_gmul_const_p_bit(tmp_start, j)] - a_j),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(state_gmul_const_p_bit(tmp_start, j)),
                    addr_input_1: addr_a_j,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_gmul_substep,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }

            // let a_j = next_row[state_gmul_const_a_bi_bit(tmp_start, j)];
            let addr_a_j = next_row(state_gmul_const_a_bi_bit(tmp_start, j));
            // let computed_bit = xor_gen(a_j, curr_row[state_gmul_const_p_bit(tmp_start, j)]);
            xor_gen(
                vec_ops,
                addr_a_j,
                curr_row(state_gmul_const_p_bit(tmp_start, j)),
            );

            // yield_constr.constraint_transition(
            //     is_gmul_substep
            //         * (computed_bit - next_row[state_gmul_const_p_bit(tmp_start, j)]),
            // );
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(state_gmul_const_p_bit(tmp_start, j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_constraint,
                addr_input_1: addr_is_gmul_substep,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
        }
    };

    let check_sub = |vec_ops: &mut Vec<VecOpConfig>,
                     steps: &[usize],
                     a_start: usize,       // start col of src a
                     sub_bit_start: usize, // start col of tmp space for sub
                     target_next_row: bool,
                     target_bit_start: usize| {
        for sub_step in 0..7 {
            // let is_gmul_substep: P = steps
            //     .iter()
            //     .map(|x| curr_row[step_bit(*x + sub_step)])
            //     .sum();
            for r in steps {
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(step_bit(*r + sub_step)),
                    addr_input_1: if r != &steps[0] {
                        addr_is_gmul_substep
                    } else {
                        0
                    },
                    addr_output: addr_is_gmul_substep,
                    is_final_output: false,
                    op_type: VecOpType::ADD,
                    op_src: VecOpSrc::VV,
                });
            }

            check_gmul(vec_ops, sub_step, a_start, sub_bit_start, sub_bit_start + 8);
        }

        // let is_sub: P = steps.iter().map(|x| curr_row[step_bit(*x + 7)]).sum();
        for r in steps {
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(step_bit(*r + 7)),
                addr_input_1: if r != &steps[0] { addr_is_sub } else { 0 },
                addr_output: addr_is_sub,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            });
        }

        let offset = [0, 4, 5, 6, 7];
        for i in 0..8 {
            for j in 1..5 {
                if j == 1 {
                    // xor_gen(
                    //     curr_row[sub_bit_start + i],
                    //     curr_row[sub_bit_start + (i + offset[j]) % 8],
                    // )
                    xor_gen(
                        vec_ops,
                        curr_row(sub_bit_start + i),
                        curr_row(sub_bit_start + (i + offset[j]) % 8),
                    );
                } else {
                    // xor_gen(
                    //     curr_row[sbox_xor_bit(sub_bit_start, j - 2, i)],
                    //     curr_row[sub_bit_start + (i + offset[j]) % 8],
                    // )
                    xor_gen(
                        vec_ops,
                        curr_row(sbox_xor_bit(sub_bit_start, j - 2, i)),
                        curr_row(sub_bit_start + (i + offset[j]) % 8),
                    );
                };
                // yield_constr.constraint(
                //     is_sub * (computed_bit - curr_row[sbox_xor_bit(sub_bit_start, j - 1, i)]),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(sbox_xor_bit(sub_bit_start, j - 1, i)),
                    addr_input_1: addr_xor_gen,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_sub,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }

            // ^ 99
            // let bit99 = FE::from_canonical_u32(BITS99LSB[i] as u32);
            // let bit_tmp = curr_row[sbox_xor_bit(sub_bit_start, 3, i)];
            let addr_bit_tmp = curr_row(sbox_xor_bit(sub_bit_start, 3, i));
            // let computed_bit = bit_tmp + bit99 - bit_tmp * bit99.doubles();
            xor_gen(vec_ops, addr_bit_tmp, 0);
            if target_next_row {
                // yield_constr.constraint_transition(
                //     is_sub * (computed_bit - next_row[target_bit_start + i]),
                // );
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_xor_gen,
                    addr_input_1: next_row(target_bit_start + i),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_sub,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint_transition(addr_constraint));
            } else {
                // yield_constr.constraint(is_sub * (computed_bit - curr_row[target_bit_start + i]));
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_xor_gen,
                    addr_input_1: curr_row(target_bit_start + i),
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: addr_constraint,
                    addr_input_1: addr_is_sub,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                vec_ops.extend(yield_constr.constraint(addr_constraint));
            }
        }
    };

    // let next_is_not_padding: P = (0..NUM_STEPS - 1).map(|i| curr_row[step_bit(i)]).sum();
    for i in 0..NUM_STEPS - 1 {
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: if i != 0 { addr_next_is_not_padding } else { 0 },
            addr_input_1: next_row(step_bit(i)),
            addr_output: addr_next_is_not_padding,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
        });
    }

    // check step bits
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
    for step in 1..NUM_STEPS {
        // yield_constr.constraint_first_row(curr_row[step_bit(step)]);
        res.extend(yield_constr.constraint_first_row(curr_row(step_bit(step))));
    }

    // inc step bits when next is not padding
    for bit in 0..NUM_STEPS {
        // degree 3
        // yield_constr.constraint_transition(
        //     next_is_not_padding
        //         * (next_row[step_bit((bit + 1) % NUM_STEPS)] - curr_row[step_bit(bit)]),
        // );
        res.extend(vec![
            VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(step_bit((bit + 1) % NUM_STEPS)),
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

    // check RCON
    let mut start_row = 0;
    for i in 0..10 {
        for step in start_row..ROUND_CHANGE_STEPS[i] + 1 {
            for bit in 0..8 {
                // yield_constr.constraint(
                //     curr_row[step_bit(step)]
                //         * (curr_row[RCON_START + bit]
                //             - FE::from_canonical_u32((RCON[i] >> 24 >> bit) & 1)),
                // );
                res.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(RCON_START + bit),
                    addr_input_1: 0,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::SUB,
                    op_src: VecOpSrc::VS,
                });
                res.push(VecOpConfig {
                    vector_length: BATCH_SIZE,
                    addr_input_0: curr_row(step_bit(step)),
                    addr_input_1: addr_constraint,
                    addr_output: addr_constraint,
                    is_final_output: false,
                    op_type: VecOpType::MUL,
                    op_src: VecOpSrc::VV,
                });
                res.extend(yield_constr.constraint(addr_constraint));
            }
        }
        start_row = ROUND_CHANGE_STEPS[i] + 1;
    }

    // check add round key
    // let is_add_round_key: P = (0..11)
    //     .map(|i| curr_row[step_bit(ADD_ROUND_KEY_STEPS[i])])
    //     .sum();
    for i in 0..11 {
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(step_bit(ADD_ROUND_KEY_STEPS[i])),
            addr_input_1: if i != 0 { addr_is_add_round_key } else { 0 },
            addr_output: addr_is_add_round_key,
            is_final_output: false,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VS,
        });
    }

    for i in 0..16 {
        for bit in 0..8 {
            // let computed_bit = xor_gen(curr_row[input_i_bit(i, bit)], curr_row[key_u8_bit(i, bit)]);
            xor_gen(
                &mut res,
                curr_row(input_i_bit(i, bit)),
                curr_row(key_u8_bit(i, bit)),
            );
            // yield_constr.constraint_transition(
            //     is_add_round_key * (computed_bit - next_row[input_i_bit(i, bit)]),
            // );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(input_i_bit(i, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_add_round_key,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));
        }
    }

    // check key update
    // key 3 rot
    check_sub(
        &mut res,
        &KEY_UPDATE_STEPS,
        key_3_after_rot_start(0),
        key_3_0_bit(0),
        false,
        KEY_3_SUB + 24,
    );
    check_sub(
        &mut res,
        &KEY_UPDATE_STEPS,
        key_3_after_rot_start(1),
        key_3_1_bit(0),
        false,
        KEY_3_SUB + 16,
    );
    check_sub(
        &mut res,
        &KEY_UPDATE_STEPS,
        key_3_after_rot_start(2),
        key_3_2_bit(0),
        false,
        KEY_3_SUB + 8,
    );
    check_sub(
        &mut res,
        &KEY_UPDATE_STEPS,
        key_3_after_rot_start(3),
        key_3_3_bit(0),
        false,
        KEY_3_SUB,
    );

    // check ^ rcon
    // let is_key_xor_rcon: P = (0..10)
    //     .map(|i| curr_row[step_bit(KEY_UPDATE_STEPS[i]) + 7])
    //     .sum();
    for bit in 0..8 {
        // let computed_bit = xor_gen(curr_row[KEY_3_SUB + 24 + bit], curr_row[RCON_START + bit]);
        xor_gen(
            &mut res,
            curr_row(KEY_3_SUB + 24 + bit),
            curr_row(RCON_START + bit),
        );
        // yield_constr.constraint(is_key_xor_rcon * (computed_bit - curr_row[key_3_rconv_bit(bit)]));
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_xor_gen,
            addr_input_1: curr_row(key_3_rconv_bit(bit)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_key_xor_rcon,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(yield_constr.constraint(addr_constraint));
    }

    // transition key
    for bit in 0..24 {
        // let computed_bit = xor_gen(curr_row[KEY_3_SUB + bit], curr_row[key_i_bit(0, bit)]);
        xor_gen(
            &mut res,
            curr_row(KEY_3_SUB + bit),
            curr_row(key_i_bit(0, bit)),
        );
        // yield_constr
        //     .constraint_transition(is_key_xor_rcon * (computed_bit - next_row[key_i_bit(0, bit)]));
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_xor_gen,
            addr_input_1: next_row(key_i_bit(0, bit)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_key_xor_rcon,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }
    for bit in 0..8 {
        // let computed_bit = xor_gen(
        //     curr_row[key_3_rconv_bit(bit)],
        //     curr_row[key_i_bit(0, 24 + bit)],
        // );
        xor_gen(
            &mut res,
            curr_row(key_3_rconv_bit(bit)),
            curr_row(key_i_bit(0, 24 + bit)),
        );
        // yield_constr.constraint_transition(
        //     is_key_xor_rcon * (computed_bit - next_row[key_i_bit(0, 24 + bit)]),
        // );
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_xor_gen,
            addr_input_1: next_row(key_i_bit(0, 24 + bit)),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_is_key_xor_rcon,
            addr_input_1: addr_constraint,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
        res.extend(yield_constr.constraint_transition(addr_constraint));
    }
    for i in 1..4 {
        for bit in 0..32 {
            // let computed_bit =
            //     xor_gen(next_row[key_i_bit(i - 1, bit)], curr_row[key_i_bit(i, bit)]);
            xor_gen(
                &mut res,
                next_row(key_i_bit(i - 1, bit)),
                curr_row(key_i_bit(i, bit)),
            );
            // yield_constr.constraint_transition(
            //     is_key_xor_rcon * (computed_bit - next_row[key_i_bit(i, bit)]),
            // );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(key_i_bit(i, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_key_xor_rcon,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));
        }
    }

    // check keep key
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_is_key_xor_rcon,
        addr_input_1: 0,
        addr_output: addr_is_not_ket_xor_rcon,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    for i in 0..4 {
        for bit in 0..32 {
            // yield_constr.constraint_transition(
            //     (P::ONES - is_key_xor_rcon)
            //         * next_is_not_padding
            //         * (next_row[key_i_bit(i, bit)] - curr_row[key_i_bit(i, bit)]),
            // );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(key_i_bit(i, bit)),
                addr_input_1: curr_row(key_i_bit(i, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_next_is_not_padding,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_not_ket_xor_rcon,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));
        }
    }

    // check state sbox
    for i in 0..16 {
        check_sub(
            &mut res,
            &STATE_SBOX_STEPS,
            input_i_bit(i, 0),
            state_i_sub_start(i),
            true,
            input_i_bit(i, 0),
        );
    }

    // check state shift rows
    // let is_state_shift_rows: P = (0..10)
    //     .map(|i| curr_row[step_bit(STATE_SHIFT_ROWS_STEPS[i])])
    //     .sum();
    for i in 0..16 {
        for j in 0..8 {
            // yield_constr.constraint_transition(
            //     is_state_shift_rows
            //         * (next_row[input_i_bit(i, j)] - curr_row[input_i_bit(SHIFT_ROWS[i], j)]),
            // );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: next_row(input_i_bit(i, j)),
                addr_input_1: curr_row(input_i_bit(SHIFT_ROWS[i], j)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_shift_rows,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));
        }
    }

    // check state mix columns
    for i in 0..4 {
        for j in 0..4 {
            let b_start = input_i_bit(i * 4 + j, 0);
            for sub_step in 0..7 {
                // let is_gmul_substep: P = STATE_MIX_COLUMNS_STEPS
                //     .iter()
                //     .map(|x| curr_row[step_bit(*x + sub_step)])
                //     .sum();
                for r in &STATE_MIX_COLUMNS_STEPS {
                    res.push(VecOpConfig {
                        vector_length: BATCH_SIZE,
                        addr_input_0: curr_row(step_bit(*r + sub_step)),
                        addr_input_1: if r != &STATE_MIX_COLUMNS_STEPS[0] {
                            addr_is_gmul_substep
                        } else {
                            0
                        },
                        addr_output: addr_is_gmul_substep,
                        is_final_output: false,
                        op_type: VecOpType::ADD,
                        op_src: VecOpSrc::VV,
                    });
                }
                check_gmul_const(
                    &mut res,
                    sub_step,
                    2,
                    b_start,
                    state_i_gmul_2_start(i * 4 + j),
                );
                check_gmul_const(
                    &mut res,
                    sub_step,
                    3,
                    b_start,
                    state_i_gmul_3_start(i * 4 + j),
                );
            }
        }

        // let is_state_mix_columns = STATE_MIX_COLUMNS_STEPS
        //     .iter()
        //     .map(|x| curr_row[step_bit(*x + 7)])
        //     .sum::<P>();
        for r in &STATE_MIX_COLUMNS_STEPS {
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: curr_row(step_bit(*r + 7)),
                addr_input_1: if r != &STATE_MIX_COLUMNS_STEPS[0] {
                    addr_is_state_mix_columns
                } else {
                    0
                },
                addr_output: addr_is_state_mix_columns,
                is_final_output: false,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
            });
        }

        for bit in 0..8 {
            // s0
            // let computed_bit = xor_gen(
            //     curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i), bit)],
            //     curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 1), bit)],
            // );
            xor_gen(
                &mut res,
                curr_row(state_gmul_const_res_bit(state_i_gmul_2_start(4 * i), bit)),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_3_start(4 * i + 1),
                    bit,
                )),
            );
            // yield_constr.constraint(
            //     is_state_mix_columns * (computed_bit - curr_row[state_i_xor_start(4 * i, 0) + bit]),
            // );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i, 0) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i, 0) + bit],
            //             curr_row[input_i_bit(4 * i + 2, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i, 0) + bit),
                curr_row(input_i_bit(4 * i + 2, bit)),
            );

            //         yield_constr.constraint(
            //             is_state_mix_columns * (computed_bit - curr_row[state_i_xor_start(4 * i, 1) + bit]),
            //         );

            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i, 1) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i, 1) + bit],
            //             curr_row[input_i_bit(4 * i + 3, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i, 1) + bit),
                curr_row(input_i_bit(4 * i + 3, bit)),
            );
            //         yield_constr.constraint_transition(
            //             is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i, bit)]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(input_i_bit(4 * i, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));

            //         // s1
            //         let computed_bit = xor_gen(
            //             curr_row[input_i_bit(4 * i, bit)],
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 1), bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(input_i_bit(4 * i, bit)),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_2_start(4 * i + 1),
                    bit,
                )),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 1, 0) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 1, 0) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 1, 0) + bit],
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 2), bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 1, 0) + bit),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_3_start(4 * i + 2),
                    bit,
                )),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 1, 1) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 1, 1) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 1, 1) + bit],
            //             curr_row[input_i_bit(4 * i + 3, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 1, 1) + bit),
                curr_row(input_i_bit(4 * i + 3, bit)),
            );
            //         yield_constr.constraint_transition(
            //             is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 1, bit)]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(input_i_bit(4 * i + 1, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));

            //         // s2
            //         let computed_bit = xor_gen(
            //             curr_row[input_i_bit(4 * i, bit)],
            //             curr_row[input_i_bit(4 * i + 1, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(input_i_bit(4 * i, bit)),
                curr_row(input_i_bit(4 * i + 1, bit)),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 2, 0) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 2, 0) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 2, 0) + bit],
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 2), bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 2, 0) + bit),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_2_start(4 * i + 2),
                    bit,
                )),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 2, 1) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 2, 1) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 2, 1) + bit],
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 3), bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 2, 1) + bit),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_3_start(4 * i + 3),
                    bit,
                )),
            );
            //         yield_constr.constraint_transition(
            //             is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 2, bit)]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(input_i_bit(4 * i + 2, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));

            //         // s3
            //         let computed_bit = xor_gen(
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i), bit)],
            //             curr_row[input_i_bit(4 * i + 1, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_gmul_const_res_bit(state_i_gmul_3_start(4 * i), bit)),
                curr_row(input_i_bit(4 * i + 1, bit)),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 3, 0) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 3, 0) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 3, 0) + bit],
            //             curr_row[input_i_bit(4 * i + 2, bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 3, 0) + bit),
                curr_row(input_i_bit(4 * i + 2, bit)),
            );
            //         yield_constr.constraint(
            //             is_state_mix_columns
            //                 * (computed_bit - curr_row[state_i_xor_start(4 * i + 3, 1) + bit]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: curr_row(state_i_xor_start(4 * i + 3, 1) + bit),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint(addr_constraint));

            //         let computed_bit = xor_gen(
            //             curr_row[state_i_xor_start(4 * i + 3, 1) + bit],
            //             curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 3), bit)],
            //         );
            xor_gen(
                &mut res,
                curr_row(state_i_xor_start(4 * i + 3, 1) + bit),
                curr_row(state_gmul_const_res_bit(
                    state_i_gmul_2_start(4 * i + 3),
                    bit,
                )),
            );
            //         yield_constr.constraint_transition(
            //             is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 3, bit)]),
            //         );
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_xor_gen,
                addr_input_1: next_row(input_i_bit(4 * i + 3, bit)),
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::SUB,
                op_src: VecOpSrc::VV,
            });
            res.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_is_state_mix_columns,
                addr_input_1: addr_constraint,
                addr_output: addr_constraint,
                is_final_output: false,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
            res.extend(yield_constr.constraint_transition(addr_constraint));
        }
    }

    // eval bits are bits
    for bit in 0..NUM_COLS {
        // yield_constr.constraint((P::ONES - curr_row[bit]) * curr_row[bit]);
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: curr_row(bit),
            addr_input_1: 0,
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
        });
        res.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_constraint,
            addr_input_1: curr_row(bit),
            addr_output: addr_constraint,
            is_final_output: false,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
        });
    }

    // free
    sys.mem.free("xor_gen_tmp");
    sys.mem.free("xor_gen");
    sys.mem.free("a_j_b_i");
    sys.mem.free("not_hi_bit_set");
    sys.mem.free("is_gmul_substep");
    sys.mem.free("is_sub");
    sys.mem.free("next_is_not_padding");
    sys.mem.free("is_add_round_key");
    sys.mem.free("is_key_xor_rcon");
    sys.mem.free("is_not_ket_xor_rcon");
    sys.mem.free("is_state_shift_rows");
    sys.mem.free("is_state_mix_columns");

    res
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let mut ramsim = RamConfig::new(&format!("{}", "aes_starky"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(32, 4096);
    let mut sys = System::new(mem, ramsim);

    let mut gene = AesTraceGenerator::<F>::new(256);
    let key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let plaintext = [0u8; 16];
    let output = gene.gen_aes(plaintext, key);
    println!("aes output: {:?}", output);

    let trace = gene.into_polynomial_values();

    let config = StarkConfig::standard_fast_config();
    let stark = S::new();
    prove::<F, C, S, D>(&mut sys, stark, &config, trace, &[], eval_packed_generic);

    info!("Total number of mem reqs: {}", sys.ramsim.op_cnt);
    info!("Total computations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}
