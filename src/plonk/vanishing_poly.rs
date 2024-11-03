use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};
use crate::memory::memory_allocator::MemAlloc;
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::plonk::zero_poly_coset::ZeroPolyOnCoset;
use crate::system::system::System;
use crate::util::{ceil_div_usize, SIZE_F};
use log::debug;
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_data::CommonCircuitData;

use crate::plonk::gates::eval_filtered_base_batch;
pub fn eval_vanishing_poly_base_batch<F: RichField + Extendable<D>, const D: usize>(
    sys: &mut System,
    common_data: &CommonCircuitData<F, D>,
    addr_xs_batch: usize, // load in previous transpose kernel
    vars_batch: EvaluationVarsBaseBatch,
    addr_local_zs: usize,         // load in previous transpose kernel
    addr_next_zs: usize,          // load in previous transpose kernel
    addr_partial_products: usize, // load in previous transpose kernel
    addr_s_sigmas: usize,         // load in previous transpose kernel
    addr_betas: usize,
    addr_gammas: usize,
    addr_alphas: usize,
    z_h_on_coset: &ZeroPolyOnCoset,
    addr_res_batch: usize, // n * num_chanllengers * SIZE_F

    eval_gate_ops: &mut VectorChain,
    eval_gate_ops_flag: &mut bool,
) {
    // debug!("eval_vanishing_poly_base_batch");

    let n = vars_batch.len();

    let max_degree = common_data.quotient_degree_factor;
    let num_prods = common_data.num_partial_products;

    let num_gate_constraints = common_data.num_gate_constraints;

    let addr_constraint_terms_batch = sys
        .mem
        .alloc("constraint_terms_batch", n * num_gate_constraints * SIZE_F)
        .unwrap();
    sys.mem
        .preload(addr_constraint_terms_batch, n * num_gate_constraints);
    if *eval_gate_ops_flag == false {
        debug!("get vec chain evaluate_gate_constraints_base_batch");
        let vec_ops = evaluate_gate_constraints_base_batch::<F, D>(
            sys,
            common_data,
            vars_batch,
            addr_constraint_terms_batch,
        );
        *eval_gate_ops = VectorChain::new(vec_ops, &sys.mem);
        *eval_gate_ops_flag = true;
        debug!("end vec chain evaluate_gate_constraints_base_batch");
    }
    sys.run_once(eval_gate_ops);
    sys.mem.unpreload(vars_batch.addr_local_constants);

    let num_challenges = common_data.config.num_challenges;
    let num_routed_wires = common_data.config.num_routed_wires;

    let addr_vanishing_z_1_terms = sys
        .mem
        .alloc("vanishing_z_1_terms", n * num_challenges * SIZE_F)
        .unwrap();
    sys.mem
        .preload(addr_vanishing_z_1_terms, n * num_challenges);
    let addr_vanishing_partial_products_terms = sys
        .mem
        .alloc(
            "vanishing_partial_products_terms",
            num_challenges * n * (num_prods + 1) * SIZE_F,
        )
        .unwrap();
    sys.mem.preload(
        addr_vanishing_partial_products_terms,
        num_challenges * n * (num_prods + 1),
    );
    let addr_l_0_x = sys.mem.alloc("l_0_x", n * SIZE_F).unwrap();
    z_h_on_coset.eval_l_0(sys, n, addr_xs_batch, addr_l_0_x);

    let mut vec_ops = Vec::new();

    let addr_k_is = sys.mem.get_addr("k_is_cpu").unwrap();
    let addr_numerator_values = sys
        .mem
        .alloc("numerator_values", num_routed_wires * n * SIZE_F)
        .unwrap();
    sys.mem.preload(addr_numerator_values, num_routed_wires * n);
    let addr_denominator_values = sys
        .mem
        .alloc("denominator_values", num_routed_wires * n * SIZE_F)
        .unwrap();
    sys.mem
        .preload(addr_denominator_values, num_routed_wires * n);

    for i in 0..num_challenges {
        // z_x.sub_one()
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_local_zs + i * n * SIZE_F,
            addr_input_1: 0,
            addr_output: addr_vanishing_z_1_terms + i * n * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VS,
            is_final_output: false,
        });
        // l_0_x * _
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_l_0_x,
            addr_input_1: addr_vanishing_z_1_terms + i * n * SIZE_F,
            addr_output: addr_vanishing_z_1_terms + i * n * SIZE_F,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        for j in 0..num_routed_wires {
            // k_i * x
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_xs_batch,
                addr_input_1: addr_k_is + j * SIZE_F,
                addr_output: addr_numerator_values + j * n * SIZE_F,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            // betas[i] * _
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_numerator_values + j * n * SIZE_F,
                addr_input_1: addr_betas + i * SIZE_F,
                addr_output: addr_numerator_values + j * n * SIZE_F,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            // wire_value + _
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: vars_batch.addr_local_wires + j * n * SIZE_F,
                addr_input_1: addr_numerator_values + j * n * SIZE_F,
                addr_output: addr_numerator_values + j * n * SIZE_F,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            // _ + gammas[i]
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_numerator_values + j * n * SIZE_F,
                addr_input_1: addr_gammas + i * SIZE_F,
                addr_output: addr_numerator_values + j * n * SIZE_F,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });

            // betas[i] * s_sigma
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_s_sigmas + j * n * SIZE_F,
                addr_input_1: addr_betas + i * SIZE_F,
                addr_output: addr_denominator_values + j * n * SIZE_F,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            // wire_value + _
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: vars_batch.addr_local_wires + j * n * SIZE_F,
                addr_input_1: addr_denominator_values + j * n * SIZE_F,
                addr_output: 0, // addr_denominator_values + j * n * SIZE_F,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            // _ + gammas[i]
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_denominator_values + j * n * SIZE_F,
                addr_input_1: addr_gammas + i * SIZE_F,
                addr_output: 0, //addr_denominator_values + j * n * SIZE_F,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
        }
        let addr_current_partial_products = addr_partial_products + i * num_prods * n * SIZE_F;
        vec_ops.extend(check_partial_products(
            &mut sys.mem,
            n,
            num_routed_wires,
            addr_numerator_values,
            addr_denominator_values,
            addr_current_partial_products,
            addr_local_zs + i * n * SIZE_F,
            addr_next_zs + i * n * SIZE_F,
            max_degree,
            addr_vanishing_partial_products_terms + i * n * (num_prods + 1) * SIZE_F,
        ))
    }
    let mut addr_vanishing_terms = (0..num_challenges)
        .map(|i| addr_vanishing_z_1_terms + i * n * SIZE_F)
        .collect::<Vec<_>>();
    addr_vanishing_terms.extend(
        (0..num_challenges * (num_prods + 1))
            .map(|i| addr_vanishing_partial_products_terms + i * n * SIZE_F)
            .collect::<Vec<_>>(),
    );
    addr_vanishing_terms.extend(
        (0..num_gate_constraints)
            .map(|i| addr_constraint_terms_batch + i * n * SIZE_F)
            .collect::<Vec<_>>(),
    );
    let terms_len = addr_vanishing_terms.len();
    for i in 0..num_challenges {
        for j in 0..terms_len {
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: if j != 0 {
                    addr_res_batch + i * n * SIZE_F
                } else {
                    0
                },
                addr_input_1: addr_alphas + i * SIZE_F,
                addr_output: addr_res_batch + i * n * SIZE_F,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_vanishing_terms[j],
                addr_input_1: addr_res_batch + i * n * SIZE_F,
                addr_output: addr_res_batch + i * n * SIZE_F,
                op_type: VecOpType::ADD,
                op_src: VecOpSrc::VV,
                is_final_output: true,
            });
        }
    }
    sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
    sys.mem.free("denominator_values");
    sys.mem.free("numerator_values");
    sys.mem.free("vanishing_partial_products_terms");
    sys.mem.free("vanishing_z_1_terms");
    sys.mem.free("l_0_x");
    sys.mem.free("constraint_terms_batch");
}

pub fn check_partial_products(
    mem: &mut MemAlloc,
    n: usize,
    num_routed_wires: usize,
    addr_numerators: usize,   // n * num_routed_wires * SIZE_F
    addr_denominators: usize, // n * num_routed_wires * SIZE_F
    addr_partials: usize,     // n * num_prods * SIZE_F
    addr_z_x: usize,          // n * SIZE_F
    addr_z_gx: usize,         // n * SIZE_F
    max_degree: usize,
    addr_res: usize, // n * (num_prods + 1) * SIZE_F
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();

    let num_prods = ceil_div_usize(num_routed_wires, max_degree) - 1;
    let mut addr_partials_split = (0..num_prods)
        .map(|i| addr_partials + i * n * SIZE_F)
        .collect::<Vec<_>>();
    addr_partials_split.insert(0, addr_z_x);
    addr_partials_split.push(addr_z_gx);

    for chunk_idx in 0..=num_prods {
        let addr_num_chunk_product = mem.alloc("num_chunk_product", n * SIZE_F).unwrap();
        let addr_den_chunk_product = mem.alloc("den_chunk_product", n * SIZE_F).unwrap();
        for i in chunk_idx * max_degree..num_routed_wires.min((chunk_idx + 1) * max_degree) {
            let addr_numerator = addr_numerators + i * n * SIZE_F;
            let addr_denominator = addr_denominators + i * n * SIZE_F;

            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_numerator,
                addr_input_1: if i != chunk_idx * max_degree {
                    addr_num_chunk_product
                } else {
                    0
                },
                addr_output: addr_num_chunk_product,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
            vec_ops.push(VecOpConfig {
                vector_length: n,
                addr_input_0: addr_denominator,
                addr_input_1: if i != chunk_idx * max_degree {
                    addr_den_chunk_product
                } else {
                    0
                },
                addr_output: addr_den_chunk_product,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
                is_final_output: false,
            });
        }
        let addr_prev_acc = addr_partials_split[chunk_idx];
        let addr_next_acc = addr_partials_split[chunk_idx + 1];
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_num_chunk_product,
            addr_input_1: addr_prev_acc,
            addr_output: addr_num_chunk_product,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_den_chunk_product,
            addr_input_1: addr_next_acc,
            addr_output: addr_den_chunk_product,
            op_type: VecOpType::MUL,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        vec_ops.push(VecOpConfig {
            vector_length: n,
            addr_input_0: addr_num_chunk_product,
            addr_input_1: addr_den_chunk_product,
            addr_output: addr_res + chunk_idx * n * SIZE_F,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
            is_final_output: false,
        });
        mem.free("num_chunk_product");
        mem.free("den_chunk_product");
    }

    vec_ops
}

pub fn evaluate_gate_constraints_base_batch<F: RichField + Extendable<D>, const D: usize>(
    sys: &mut System,
    common_data: &CommonCircuitData<F, D>,
    vars_batch: EvaluationVarsBaseBatch,
    addr_constraints_batch: usize,
) -> Vec<VecOpConfig> {
    debug!("evaluate_gate_constraints_base_batch");

    let mut res = Vec::new();
    for (i, gate) in common_data.gates.iter().enumerate() {
        let selector_index = common_data.selectors_info.selector_indices[i];

        let addr_res_batch = sys
            .mem
            .alloc(
                "res_batch",
                vars_batch.len() * gate.0.num_constraints() * SIZE_F,
            )
            .unwrap();

        let group_id = format!("group_{}_cpu", selector_index);
        let addr_group = sys.mem.get_addr(&group_id).unwrap();
        let group_size = sys.mem.get_size(&group_id).unwrap() / SIZE_F;
        if !sys.mem.preloaded(addr_group) {
            sys.mem.preload(addr_group, group_size);

            let mut group_cp = MemCpy::new(
                MemCpyConfig {
                    addr_input: addr_group,
                    addr_output: addr_group,
                    input_length: group_size,
                },
                unsafe { ENABLE_CONFIG.other },
            );
            group_cp.drain.clear();
            group_cp.write_request.clear();
            sys.run_once(&group_cp);
        }

        let mut gate_eval_ops = eval_filtered_base_batch(
            sys,
            gate,
            vars_batch,
            i,
            selector_index,
            addr_group,
            common_data.selectors_info.groups[selector_index].clone(),
            common_data.selectors_info.num_selectors(),
            0,
            addr_res_batch,
        );
        gate_eval_ops.push(VecOpConfig {
            vector_length: vars_batch.len() * gate.0.num_constraints(),
            addr_input_0: addr_constraints_batch,
            addr_input_1: addr_res_batch,
            addr_output: addr_constraints_batch,
            op_type: VecOpType::ADD,
            op_src: VecOpSrc::VV,
            is_final_output: true,
        });
        res.extend(gate_eval_ops);

        sys.mem.free("res_batch");
    }
    res
}
