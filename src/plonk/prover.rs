use log::info;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::fri::FriParams;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::PartitionWitness;
use plonky2::plonk::circuit_data::{CommonCircuitData, ProverOnlyCircuitData};
use plonky2::plonk::config::GenericConfig;
use plonky2::util::{log2_ceil, log2_strict};
use rand::Rng;

use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::fft::{Fft, FftConfig, FftDirection};
use crate::kernel::hash_no_pad::{HashNoPad, HashNoPadConfig};
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::transpose::{Transpose, TransposeConfig};
use crate::kernel::tree::{Tree, TreeConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpConfig, VecOpExtension, VecOpSrc, VecOpType};
use crate::plonk::challenger::Challenger;
use crate::plonk::oracle::PolynomialBatch;
use crate::plonk::proofs::OpeningSet;
use crate::plonk::vanishing_poly::eval_vanishing_poly_base_batch;
use crate::plonk::vars::EvaluationVarsBaseBatch;
use crate::plonk::zero_poly_coset::ZeroPolyOnCoset;
use crate::system::system::System;
use crate::util::{ceil_div_usize, log2, BATCH_SIZE, NUM_HASH_OUT_ELTS, SIZE_F, SPONGE_RATE};

/// "*_cpu" indicates that the data is computed on the CPU
pub fn prove_with_partition_witness<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    const D: usize,
>(
    sys: &mut System,
    prover_data: &ProverOnlyCircuitData<F, C, D>,
    common_data: &CommonCircuitData<F, D>,
    partition_witness: PartitionWitness<F>,
) {
    unsafe {
        info!("ArchConfig is {:?}", ARCH_CONFIG);
    }
    info!("Proving with partition witness");

    let config = &common_data.config;
    let num_challenges = config.num_challenges;
    let degree = common_data.degree();

    let addr_public_inputs = sys
        .mem
        .alloc(
            "public_inputs_cpu",
            prover_data.public_inputs.len() * SIZE_F,
        )
        .unwrap(); // we don't support get_targets operation, should be done in the CPU
    let addr_public_inputs_hash = sys
        .mem
        .alloc("public_inputs_hash", NUM_HASH_OUT_ELTS * SIZE_F)
        .unwrap();
    let public_inputs_hash_kernel = HashNoPad::new(HashNoPadConfig {
        addr_input: vec![addr_public_inputs],
        addr_output: addr_public_inputs_hash,
        input_length: vec![prover_data.public_inputs.len()],
        output_length: NUM_HASH_OUT_ELTS,
    });
    sys.run_once(&public_inputs_hash_kernel);

    let addr_witness = sys
        .mem
        .alloc(
            "witness_cpu",
            partition_witness.num_wires * partition_witness.degree * SIZE_F,
        )
        .unwrap();

    let wires_commitment = PolynomialBatch::new(
        "wires_commitment",
        sys,
        addr_witness,
        partition_witness.degree,
        partition_witness.num_wires,
        config.fri_config.rate_bits,
        config.zero_knowledge,
        config.fri_config.cap_height,
        true,
        Vec::new(),
    );

    let addr_chanllenger_output_buffer = sys
        .mem
        .alloc("Challenger.output_buffer", SPONGE_RATE * SIZE_F)
        .unwrap();
    let addr_prover_data_circuit_digest = sys
        .mem
        .alloc("prover_data.circuit_digest_cpu", NUM_HASH_OUT_ELTS * SIZE_F)
        .unwrap();

    let mut challenger = Challenger::new(addr_chanllenger_output_buffer);
    challenger.observe_hash(sys, addr_prover_data_circuit_digest);
    challenger.observe_hash(sys, addr_public_inputs_hash);

    challenger.observe_cap(sys, wires_commitment.addr_cap, config.fri_config.cap_height);

    let addr_betas = sys.mem.alloc("betas", num_challenges * SIZE_F).unwrap();
    let addr_gammas = sys.mem.alloc("gammas", num_challenges * SIZE_F).unwrap();
    challenger.get_n_challenges(sys, addr_betas, num_challenges);
    challenger.get_n_challenges(sys, addr_gammas, num_challenges);

    let num_routed_wires_quotient = common_data.num_partial_products + 1;
    let addr_sigmas = sys
        .mem
        .alloc(
            "sigmas_cpu",
            common_data.config.num_routed_wires * prover_data.subgroup.len() * SIZE_F,
        )
        .unwrap();
    let addr_subgroup = sys
        .mem
        .alloc("subgroup_cpu", prover_data.subgroup.len() * SIZE_F)
        .unwrap();
    let addr_k_is = sys
        .mem
        .alloc("k_is_cpu", common_data.config.num_routed_wires * SIZE_F)
        .unwrap();
    let addr_partial_products_and_zs = sys
        .mem
        .alloc(
            "partial_products_and_zs",
            num_challenges * num_routed_wires_quotient * prover_data.subgroup.len() * SIZE_F,
        )
        .unwrap(); // num_challenges Matrix of num_routed_wires_quotient x subgroup.len()

    all_wires_permutation_partial_products(
        sys,
        addr_partial_products_and_zs,
        addr_witness,
        addr_betas,
        addr_gammas,
        addr_sigmas,
        addr_subgroup,
        prover_data.subgroup.len(),
        addr_k_is,
        common_data.config.num_routed_wires,
        partition_witness.num_wires,
        common_data.quotient_degree_factor,
        num_challenges,
    );

    let addr_zs_partial_products = sys
        .mem
        .alloc(
            "zs_partial_products",
            num_challenges * num_routed_wires_quotient * prover_data.subgroup.len() * SIZE_F,
        )
        .unwrap(); // num_challenges * num_routed_wires_quotient polynomials of degree prover_data.subgroup.len()

    let mut mks = Vec::new();
    for i in 0..num_challenges {
        let mk = MemCpy::new(
            MemCpyConfig {
                addr_input: addr_partial_products_and_zs
                    + i * num_routed_wires_quotient * prover_data.subgroup.len() * SIZE_F,
                addr_output: addr_zs_partial_products
                    + num_challenges * prover_data.subgroup.len()
                    + i * (num_routed_wires_quotient - 1) * prover_data.subgroup.len(),
                input_length: (num_routed_wires_quotient - 1) * prover_data.subgroup.len(),
            },
            false,
        );
        mks.push(mk);

        let mk = MemCpy::new(
            MemCpyConfig {
                addr_input: addr_partial_products_and_zs
                    + i * num_routed_wires_quotient * prover_data.subgroup.len() * SIZE_F
                    + (num_routed_wires_quotient - 1) * prover_data.subgroup.len() * SIZE_F,
                addr_output: addr_zs_partial_products + i * prover_data.subgroup.len() * SIZE_F,
                input_length: prover_data.subgroup.len(),
            },
            false,
        );
        mks.push(mk);
    }

    sys.mem.free("partial_products_and_zs");
    let partial_products_zs_and_lookup_commitment = PolynomialBatch::new(
        "partial_products_zs_and_lookup_commitment",
        sys,
        addr_zs_partial_products,
        prover_data.subgroup.len(),
        num_challenges * num_routed_wires_quotient,
        config.fri_config.rate_bits,
        config.zero_knowledge,
        config.fri_config.cap_height,
        false,
        mks,
    );

    sys.mem.free("zs_partial_products");

    challenger.observe_cap(
        sys,
        partial_products_zs_and_lookup_commitment.addr_cap,
        config.fri_config.cap_height,
    );

    let addr_alphas = sys.mem.alloc("alphas", num_challenges * SIZE_F).unwrap();
    challenger.get_n_challenges(sys, addr_alphas, num_challenges);

    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let addr_points = sys
        .mem
        .alloc(
            "points_cpu",
            (1 << (common_data.degree_bits() + quotient_degree_bits)) * SIZE_F,
        )
        .unwrap();

    let constants_sigmas_commitment = PolynomialBatch::new_alloc(
        "constants_sigmas_commitment",
        sys,
        1 << prover_data.constants_sigmas_commitment.degree_log,
        prover_data.constants_sigmas_commitment.merkle_tree.leaves[0].len(),
        prover_data.constants_sigmas_commitment.rate_bits,
        prover_data.constants_sigmas_commitment.blinding,
        log2_strict(
            prover_data
                .constants_sigmas_commitment
                .merkle_tree
                .cap
                .len(),
        ),
    ); // copy from cpu

    for (i, group) in common_data.selectors_info.groups.iter().enumerate() {
        let group_id = format!("group_{}_cpu", i);
        let group_length = group.len();
        sys.mem.alloc(&group_id, group_length * SIZE_F);
    }
    let lde_size = 1 << (common_data.degree_bits() + quotient_degree_bits);

    let addr_quotient_polys = sys
        .mem
        .alloc("quotient_polys", num_challenges * lde_size * SIZE_F)
        .unwrap();

    compute_quotient_polys(
        sys,
        common_data,
        addr_points,
        addr_public_inputs_hash,
        &constants_sigmas_commitment,
        &wires_commitment,
        &partial_products_zs_and_lookup_commitment,
        addr_betas,
        addr_gammas,
        addr_alphas,
        addr_quotient_polys,
    );
    let quotient_polys_commitment = PolynomialBatch::new(
        "quotient_polys_commitment",
        sys,
        addr_quotient_polys,
        degree,
        num_challenges * common_data.quotient_degree_factor,
        config.fri_config.rate_bits,
        config.zero_knowledge,
        config.fri_config.cap_height,
        false,
        Vec::new(),
    );
    sys.mem.free("quotient_polys");

    challenger.observe_cap(
        sys,
        quotient_polys_commitment.addr_cap,
        config.fri_config.cap_height,
    );
    let addr_zeta = sys.mem.alloc("zeta", D * SIZE_F).unwrap();
    let addr_zeta_g = sys.mem.alloc("zeta_g", D * SIZE_F).unwrap();
    challenger.get_n_challenges(sys, addr_zeta, D);
    let zeta_g_kernel = VectorChain::new(
        VecOpExtension::<D>::mul(
            &mut sys.mem,
            1,
            addr_zeta,
            0,
            addr_zeta_g,
            VecOpSrc::VS,
            true,
        ),
        &sys.mem,
    );
    sys.run_once(&zeta_g_kernel);

    let openings = OpeningSet::new(
        sys,
        addr_zeta,
        addr_zeta_g,
        &constants_sigmas_commitment,
        &wires_commitment,
        &partial_products_zs_and_lookup_commitment,
        &quotient_polys_commitment,
        common_data,
    );

    challenger.observe_openings(sys, &openings);

    let instance = common_data.get_fri_instance(F::Extension::ONE);
    PolynomialBatch::prove_openings(
        sys,
        &instance,
        &vec![
            &constants_sigmas_commitment,
            &wires_commitment,
            &partial_products_zs_and_lookup_commitment,
            &quotient_polys_commitment,
        ],
        &mut challenger,
        &common_data.fri_params,
        addr_zeta,
        addr_zeta_g,
    );
}

fn all_wires_permutation_partial_products(
    sys: &mut System,
    addr_output: usize,
    addr_witness: usize,
    addr_betas: usize,
    addr_gammas: usize,
    addr_sigmas: usize,
    addr_subgroup: usize,
    subgroup_length: usize,
    addr_k_is: usize,
    num_routed_wires: usize,
    num_wires: usize,
    degree: usize,
    num_challenges: usize,
) {
    info!("all_wires_permutation_partial_products");
    (0..num_challenges).for_each(|i| {
        wires_permutation_partial_products_and_zs(
            sys,
            addr_output + i * num_routed_wires * subgroup_length * SIZE_F,
            addr_witness,
            addr_betas + i * SIZE_F,
            addr_gammas + i * SIZE_F,
            addr_sigmas,
            addr_subgroup,
            subgroup_length,
            addr_k_is,
            num_routed_wires,
            num_wires,
            degree,
        )
    });
}

pub fn wires_permutation_partial_products_and_zs(
    sys: &mut System,
    addr_output: usize,
    addr_witness: usize,
    addr_beta: usize,
    addr_gamma: usize,
    addr_sigmas: usize,
    addr_subgroup: usize,
    subgroup_length: usize,
    addr_k_is: usize,
    num_routed_wires: usize,
    num_wires: usize,
    degree: usize,
) {
    info!("wires_permutation_partial_products_and_zs");

    let num_routed_wires_quotient = ceil_div_usize(num_routed_wires, degree);

    let addr_all_quotient_chunk_products = sys
        .mem
        .alloc(
            "all_quotient_chunk_products",
            subgroup_length * num_routed_wires_quotient * SIZE_F,
        )
        .unwrap();
    let addr_numerators = sys
        .mem
        .alloc("numerators", subgroup_length * num_routed_wires * SIZE_F)
        .unwrap();
    let addr_denominators = sys
        .mem
        .alloc("denominators", subgroup_length * num_routed_wires * SIZE_F)
        .unwrap();
    let addr_quotient_values = sys
        .mem
        .alloc(
            "quotient_values",
            subgroup_length * num_routed_wires * SIZE_F,
        )
        .unwrap();

    sys.mem.preload(addr_k_is, num_routed_wires);
    sys.mem.preload(addr_beta, 1);
    sys.mem.preload(addr_gamma, 1);

    let mut vec_ops = Vec::new();
    for i in 0..subgroup_length {
        // let s_id = k_i * x;
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_k_is,
            addr_input_1: addr_subgroup + i * SIZE_F,
            addr_output: addr_numerators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VS,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });
        // beta * s_id
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_numerators + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_beta,
            addr_output: addr_numerators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VS,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });
        // wire_value + _
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_witness + i * num_wires * SIZE_F,
            addr_input_1: addr_numerators + i * num_routed_wires * SIZE_F,
            addr_output: addr_numerators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VV,
            op_type: VecOpType::ADD,
            is_final_output: false,
        });
        // _ + gamma
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_numerators + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_gamma,
            addr_output: addr_numerators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VS,
            op_type: VecOpType::ADD,
            is_final_output: false,
        });

        // beta * s_sigma
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_sigmas + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_beta,
            addr_output: addr_denominators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VS,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });

        // wire_value + _
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_witness + i * num_wires * SIZE_F,
            addr_input_1: addr_denominators + i * num_routed_wires * SIZE_F,
            addr_output: addr_denominators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VV,
            op_type: VecOpType::ADD,
            is_final_output: false,
        });

        // _ + gamma
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_denominators + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_gamma,
            addr_output: addr_denominators + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VS,
            op_type: VecOpType::ADD,
            is_final_output: false,
        });

        // denominator_invs
        vec_ops.extend(VecOpConfig::inv(
            &mut sys.mem,
            num_routed_wires,
            addr_denominators + i * num_routed_wires * SIZE_F,
            addr_denominators + i * num_routed_wires * SIZE_F,
            false,
        ));

        // quotient_values
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_numerators + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_denominators + i * num_routed_wires * SIZE_F,
            addr_output: addr_quotient_values + i * num_routed_wires * SIZE_F,
            op_src: VecOpSrc::VV,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });

        // apply systolic array for QCP, only one more mul
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires,
            addr_input_0: addr_quotient_values + i * num_routed_wires * SIZE_F,
            addr_input_1: addr_quotient_values + i * num_routed_wires * SIZE_F,
            addr_output: addr_all_quotient_chunk_products + i * num_routed_wires_quotient * SIZE_F,
            op_src: VecOpSrc::VV,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });

        // PP
        vec_ops.push(VecOpConfig {
            vector_length: num_routed_wires_quotient,
            addr_input_0: addr_all_quotient_chunk_products + i * num_routed_wires_quotient * SIZE_F,
            addr_input_1: addr_all_quotient_chunk_products + i * num_routed_wires_quotient * SIZE_F,
            addr_output: addr_all_quotient_chunk_products + i * num_routed_wires_quotient * SIZE_F,
            op_src: VecOpSrc::VV,
            op_type: VecOpType::MUL,
            is_final_output: false,
        });

        if i > 0 {
            vec_ops.push(VecOpConfig {
                vector_length: num_routed_wires_quotient,
                addr_input_0: addr_all_quotient_chunk_products
                    + i * num_routed_wires_quotient * SIZE_F,
                addr_input_1: addr_all_quotient_chunk_products
                    + (i - 1) * num_routed_wires_quotient * SIZE_F,
                addr_output: addr_output + i * num_routed_wires_quotient * SIZE_F,
                op_src: VecOpSrc::VS,
                op_type: VecOpType::MUL,
                is_final_output: true,
            });
        }
    }
    sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
    sys.mem.clear_preload();

    sys.mem.free("numerators");
    sys.mem.free("denominators");
    sys.mem.free("quotient_values");

    sys.mem.free("all_quotient_chunk_products");
}

fn compute_quotient_polys<F: RichField + Extendable<D>, const D: usize>(
    sys: &mut System,

    common_data: &CommonCircuitData<F, D>,

    addr_points: usize,
    addr_public_inputs_hash: usize,
    constants_sigmas_commitment: &PolynomialBatch,
    wires_commitment: &PolynomialBatch,
    zs_partial_products_and_lookup_commitment: &PolynomialBatch,
    addr_betas: usize,
    addr_gammas: usize,
    addr_alphas: usize,
    addr_res: usize,
) {
    info!("compute_quotient_polys");
    let num_challenges = common_data.config.num_challenges;
    let rate_bits = common_data.config.fri_config.rate_bits;
    let degree_bits = common_data.degree_bits();
    let quotient_degree_bits = log2_ceil(common_data.quotient_degree_factor);
    let num_routed_wires = common_data.config.num_routed_wires;
    let num_wires = common_data.config.num_wires;
    let num_constants = common_data.config.num_constants;
    let num_partial_products = common_data.num_partial_products;

    let step = 1 << (rate_bits - quotient_degree_bits);
    let next_step = 1 << quotient_degree_bits;
    let lde_size = 1 << (degree_bits + quotient_degree_bits);
    println!("lde_size: {}", lde_size);
    let num_batches = ceil_div_usize(lde_size, BATCH_SIZE);

    let z_h_on_coset = ZeroPolyOnCoset::new(sys, rate_bits);

    let addr_res_tmp = sys
        .mem
        .alloc("quotient_values_tmp", num_challenges * lde_size * SIZE_F)
        .unwrap();

    let mut mks = Vec::new();

    let mut eval_gate_ops = VectorChain::new(vec![], &sys.mem);
    let mut eval_gate_flag = false;

    let addr_k_is = sys.mem.get_addr("k_is_cpu").unwrap();
    let mut k_is_cp = MemCpy::new(
        MemCpyConfig {
            addr_input: addr_k_is,
            addr_output: 0,
            input_length: num_routed_wires,
        },
        unsafe { ENABLE_CONFIG.poly },
    );
    k_is_cp.drain.clear();
    k_is_cp.write_request.clear();
    sys.run_once(&k_is_cp);

    info!("num_batches: {}", num_batches);
    for batch_i in 0..num_batches {
        if batch_i % 256 == 0 {
            info!("batch_i: {}", batch_i);
        }

        sys.mem.preload(addr_k_is, num_routed_wires);

        let xs_batch_len = BATCH_SIZE.min(lde_size - batch_i * BATCH_SIZE);
        let indices_batch: Vec<usize> =
            (BATCH_SIZE * batch_i..BATCH_SIZE * batch_i + xs_batch_len).collect();

        let mut mcks = Vec::new();

        let addr_shifted_xs_batch = sys
            .mem
            .alloc("shifted_xs_batch", xs_batch_len * SIZE_F)
            .unwrap();
        let mut xs_batch_cp = MemCpy::new(
            MemCpyConfig {
                addr_input: addr_points + batch_i * BATCH_SIZE * SIZE_F,
                addr_output: addr_shifted_xs_batch,
                input_length: xs_batch_len,
            },
            unsafe { ENABLE_CONFIG.poly },
        );
        xs_batch_cp.drain.clear();
        mcks.push(xs_batch_cp);
        sys.mem.preload(addr_shifted_xs_batch, xs_batch_len);

        let mut addr_local_constants_sigmas_batch = Vec::with_capacity(xs_batch_len);
        let mut addr_local_zs_batch = Vec::with_capacity(xs_batch_len);
        let mut addr_next_zs_batch = Vec::with_capacity(xs_batch_len);
        let mut addr_partial_products_batch = Vec::with_capacity(xs_batch_len);
        let mut addr_s_sigmas_batch = Vec::with_capacity(xs_batch_len);
        let mut addr_local_constants_batch_refs = Vec::with_capacity(xs_batch_len);
        let mut addr_local_wires_batch_refs = Vec::with_capacity(xs_batch_len);
        let mut addr_local_zs_partial_and_lookup = Vec::with_capacity(xs_batch_len);

        for i in indices_batch {
            let i_next = (i + next_step) % lde_size;
            let mut local_constants_sigmas =
                constants_sigmas_commitment.get_lde_values_addr(i, step);
            local_constants_sigmas.1 = num_constants + num_routed_wires;
            let local_constants = (local_constants_sigmas.0, num_constants);
            let s_sigmas = (
                local_constants_sigmas.0 + num_constants * SIZE_F,
                num_routed_wires,
            );
            let local_wires = wires_commitment.get_lde_values_addr(i, step);
            let mut local_zs_partial_and_lookup =
                zs_partial_products_and_lookup_commitment.get_lde_values_addr(i, step);
            local_zs_partial_and_lookup.1 = (num_partial_products + 1) * num_challenges;

            let next_zs_partial_and_lookup =
                zs_partial_products_and_lookup_commitment.get_lde_values_addr(i_next, step);

            let local_zs = (local_zs_partial_and_lookup.0, num_challenges);
            let next_zs = (next_zs_partial_and_lookup.0, num_challenges);

            let partial_products = (
                local_zs_partial_and_lookup.0 + num_challenges * SIZE_F,
                num_partial_products * num_challenges,
            );

            assert_eq!(local_wires.1, num_wires);
            assert_eq!(local_zs.1, num_challenges);

            addr_local_constants_sigmas_batch.push(local_constants_sigmas);

            addr_local_constants_batch_refs.push(local_constants);
            addr_local_wires_batch_refs.push(local_wires);

            addr_local_zs_partial_and_lookup.push(local_zs_partial_and_lookup);
            addr_local_zs_batch.push(local_zs);
            addr_next_zs_batch.push(next_zs);
            addr_partial_products_batch.push(partial_products);
            addr_s_sigmas_batch.push(s_sigmas);
        }
        // println!("addr_local_constants_sigmas_batch[0]: {:?}", addr_local_constants_sigmas_batch[0]);
        // println!("addr_local_wires_batch_refs[0]: {:?}", addr_local_wires_batch_refs[0]);
        // println!("addr_local_zs_partial_and_lookup[0]: {:?}", addr_local_zs_partial_and_lookup[0]);
        // println!("addr_next_zs_batch[0]: {:?}", addr_next_zs_batch[0]);

        mcks.extend(batch_copy_transpose(&addr_local_constants_sigmas_batch));
        mcks.extend(batch_copy_transpose(&addr_local_wires_batch_refs));
        mcks.extend(batch_copy_transpose(&addr_local_zs_partial_and_lookup));
        mcks.extend(batch_copy_transpose(&addr_next_zs_batch));

        let addr_local_constants_batch = sys
            .mem
            .alloc(
                "local_constants_batch",
                xs_batch_len * addr_local_constants_batch_refs[0].1 * SIZE_F,
            )
            .unwrap();
        sys.mem.preload(
            addr_local_constants_batch,
            xs_batch_len * addr_local_constants_batch_refs[0].1,
        );

        let addr_local_wires_batch = sys
            .mem
            .alloc(
                "local_wires_batch",
                xs_batch_len * addr_local_wires_batch_refs[0].1 * SIZE_F,
            )
            .unwrap();
        sys.mem.preload(
            addr_local_wires_batch,
            xs_batch_len * addr_local_wires_batch_refs[0].1,
        );

        let vars_batch = EvaluationVarsBaseBatch::new(
            xs_batch_len,
            addr_local_constants_batch,
            addr_local_wires_batch,
            addr_public_inputs_hash,
        );

        let addr_local_zs = sys
            .mem
            .alloc("local_zs", xs_batch_len * num_challenges * SIZE_F)
            .unwrap();
        sys.mem
            .preload(addr_local_zs, xs_batch_len * num_challenges);
        let addr_next_zs = sys
            .mem
            .alloc("next_zs", xs_batch_len * num_challenges * SIZE_F)
            .unwrap();
        sys.mem.preload(addr_next_zs, xs_batch_len * num_challenges);
        let addr_partial_products = sys
            .mem
            .alloc(
                "partial_products",
                xs_batch_len * addr_partial_products_batch[0].1 * SIZE_F,
            )
            .unwrap();
        sys.mem.preload(
            addr_partial_products,
            xs_batch_len * addr_partial_products_batch[0].1,
        );
        let addr_s_sigmas = sys
            .mem
            .alloc("s_sigmas", xs_batch_len * addr_s_sigmas_batch[0].1 * SIZE_F)
            .unwrap();
        sys.run_vec(mcks);
        sys.mem
            .preload(addr_s_sigmas, xs_batch_len * addr_s_sigmas_batch[0].1);

        let addr_quotient_values_batch = sys
            .mem
            .alloc(
                "quotient_values_batch",
                xs_batch_len * num_challenges * SIZE_F,
            )
            .unwrap();
        sys.mem
            .preload(addr_quotient_values_batch, xs_batch_len * num_challenges);
        z_h_on_coset.preload(&mut sys.mem);

        sys.mem.preload(addr_betas, num_challenges);
        sys.mem.preload(addr_gammas, num_challenges);
        sys.mem.preload(addr_alphas, num_challenges);

        eval_vanishing_poly_base_batch(
            sys,
            common_data,
            addr_shifted_xs_batch,
            vars_batch,
            addr_local_zs,
            addr_next_zs,
            addr_partial_products,
            addr_s_sigmas,
            addr_betas,
            addr_gammas,
            addr_alphas,
            &z_h_on_coset,
            addr_quotient_values_batch,
            &mut eval_gate_ops,
            &mut eval_gate_flag,
        );

        let mut vec_ops = Vec::new();
        for i in 0..num_challenges {
            let addr_denominator_inv = z_h_on_coset.eval_inverse(0);
            vec_ops.push(VecOpConfig {
                vector_length: xs_batch_len,
                addr_input_0: addr_quotient_values_batch + i * xs_batch_len * SIZE_F,
                addr_input_1: addr_denominator_inv,
                addr_output: addr_quotient_values_batch + i * xs_batch_len * SIZE_F,
                op_src: VecOpSrc::VV,
                op_type: VecOpType::MUL,
                is_final_output: true,
            });
        }
        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
        for i in 0..num_challenges {
            let mk = MemCpy::new(
                MemCpyConfig {
                    addr_input: addr_quotient_values_batch + i * xs_batch_len * SIZE_F,
                    addr_output: addr_res_tmp
                        + i * xs_batch_len * SIZE_F * num_batches
                        + batch_i * xs_batch_len * SIZE_F,
                    input_length: xs_batch_len,
                },
                false,
            );
            mks.push(mk);
        }

        sys.mem.clear_preload();
        sys.mem.free("local_constants_batch");
        sys.mem.free("local_wires_batch");
        sys.mem.free("local_zs");
        sys.mem.free("next_zs");
        sys.mem.free("partial_products");
        sys.mem.free("s_sigmas");
        sys.mem.free("quotient_values_batch");

        sys.mem.free("shifted_xs_batch");
    }

    let mut fft_k = Fft::new(FftConfig {
        lg_n: log2_ceil(lde_size),
        k: num_challenges,
        direction: FftDirection::NN,
        addr_input: addr_res_tmp,
        addr_tmp: 1 << 60,
        addr_output: addr_res,
        inverse: true,
        rate_bits: 0,
        coset: true,
        extension: 1,
        transposed_input: false,
    });
    for memcpy in &mks {
        fft_k.prefetch.addr_trans(memcpy);
    }
    sys.run_once(&fft_k);
    sys.mem.free("quotient_values_tmp");
}

fn batch_copy_transpose(batch_addr: &[(usize, usize)]) -> Vec<MemCpy> {
    let al = unsafe { ARCH_CONFIG.array_length };
    let width = batch_addr[0].1;
    let height = batch_addr.len();

    let mut mcs = Vec::with_capacity(height * width);
    for i_chunk in (0..height).collect::<Vec<_>>().chunks(al) {
        for i in i_chunk {
            let mut mk = MemCpy::new(
                MemCpyConfig {
                    addr_input: batch_addr[*i].0,
                    addr_output: 0,
                    input_length: width,
                },
                unsafe { ENABLE_CONFIG.poly },
            );
            mk.drain.clear();
            mk.write_request.clear();
            mcs.push(mk);
        }
    }

    mcs
}

pub fn fri_proof<const D: usize>(
    sys: &mut System,
    addr_initial_merkle_tree: &Vec<&PolynomialBatch>,
    addr_lde_polynomial_coeffs: usize,
    addr_lde_coset_values: usize,
    degree: usize,
    challenger: &mut Challenger,
    fri_params: &FriParams,
) {
    let mut addr_trees: Vec<PolynomialBatch> = Vec::new();
    fri_committed_trees::<D>(
        sys,
        addr_lde_polynomial_coeffs,
        addr_lde_coset_values,
        degree,
        challenger,
        fri_params,
        &mut addr_trees,
    );

    sys.run_once(&challenger.pow());

    fri_prover_query_rounds(
        sys,
        addr_initial_merkle_tree,
        &addr_trees,
        degree,
        fri_params,
    );
}

pub fn fri_committed_trees<const D: usize>(
    sys: &mut System,
    addr_coeffs: usize,
    addr_values: usize,
    mut degree: usize,
    challenger: &mut Challenger,
    fri_params: &FriParams,
    addr_trees: &mut Vec<PolynomialBatch>,
) {
    let cap_height = fri_params.config.cap_height;

    let addr_shift = sys.mem.alloc("shift", D * SIZE_F).unwrap();
    for (arity_i, arity_bits) in (&fri_params.reduction_arity_bits).iter().enumerate() {
        let arity = 1 << arity_bits;

        let digests_id = format!("{}{}", "digests", arity_i);
        let cap_id = format!("{}{}", "cap", arity_i);
        let leaves_id = format!("{}{}", "leaves", arity_i);

        let num_leaves = degree / arity;
        let leaf_length = arity * D;
        let addr_digests = sys
            .mem
            .alloc(
                &digests_id,
                Tree::num_digests(num_leaves, cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_cap = sys
            .mem
            .alloc(
                &cap_id,
                Tree::num_caps(cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_leaves = sys
            .mem
            .alloc(&leaves_id, num_leaves * leaf_length * SIZE_F)
            .unwrap();
        addr_trees.push(PolynomialBatch {
            name: format!("Tree_{}", arity_i),
            addr_cap: addr_cap,
            addr_digests: addr_digests,
            addr_leaves: addr_leaves,
            addr_polynomials: 0,
            addr_salt: 0,
            leaf_length: leaf_length,
            degree_log: log2(num_leaves),
            rate_bits: fri_params.config.rate_bits,
            blinding: false,
            addr_transposed_leaves: 0,
            padding_length: 0,
        });

        let mut mk = MemCpy::new(
            MemCpyConfig {
                addr_input: addr_coeffs,
                addr_output: addr_leaves,
                input_length: num_leaves * leaf_length,
            },
            unsafe { ENABLE_CONFIG.tree },
        );
        mk.prefetch.clear();
        mk.read_request.clear();
        sys.run_once(&mk);

        let tree = Tree::new(TreeConfig {
            leaf_length: leaf_length,
            cap_height: cap_height,
            num_leaves: num_leaves,
            addr_cap_buf: addr_cap,
            addr_leaves: addr_leaves,
            addr_transposed_leaves: 0,
            addr_digest_buf: addr_digests,
            transposed_leaves: false,
        });
        sys.run_once(&tree);

        challenger.observe_cap(sys, addr_cap, cap_height);

        let addr_beta = sys.mem.alloc("beta", D * SIZE_F).unwrap();
        challenger.get_extension_challenge::<D>(sys, addr_beta);

        let addr_coeffs_trans = sys.mem.alloc("coeffs_trans", degree * D * SIZE_F).unwrap();

        let chunk_length = unsafe { ARCH_CONFIG.num_elems() / arity / D };
        for chunk in (0..degree / arity).step_by(chunk_length) {
            let mut vec_ops = Vec::new();
            let mut coeffs_trans = Transpose::new(TransposeConfig {
                addr_input: addr_coeffs + chunk * arity * D * SIZE_F,
                addr_output: addr_coeffs_trans,
                width: arity,
                height: chunk_length.min(degree / arity - chunk),
                reverse: false,
                extension: D,
                start: 0,
                end: arity,
            });
            coeffs_trans.drain.clear();
            coeffs_trans.write_request.clear();
            sys.run_once(&coeffs_trans);

            for i in 0..arity {
                vec_ops.extend(VecOpExtension::<D>::mul(
                    &mut sys.mem,
                    degree / arity,
                    addr_coeffs,
                    addr_beta,
                    addr_coeffs,
                    VecOpSrc::VS,
                    false,
                ));
                vec_ops.extend(VecOpExtension::<D>::add(
                    degree / arity,
                    addr_coeffs,
                    addr_coeffs_trans + i * degree / arity * D * SIZE_F,
                    addr_coeffs,
                    VecOpSrc::VV,
                    true,
                ));
            }
            sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
        }

        let mut vec_ops = Vec::new();
        vec_ops.extend(VecOpExtension::<D>::exp_u64(
            &mut sys.mem,
            addr_shift,
            addr_shift,
            true,
        ));
        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));

        sys.run_once(&Fft::new(FftConfig {
            lg_n: log2(degree / arity),
            k: 1,
            direction: FftDirection::NR,
            addr_input: addr_coeffs,
            addr_tmp: addr_coeffs_trans,
            addr_output: addr_values,
            inverse: false,
            rate_bits: 0,
            coset: true,
            extension: D,
            transposed_input: false,
        }));
        sys.mem.free("coeffs_trans");
        sys.mem.free("beta");
        degree /= arity;
    }
    sys.mem.free("shift");

    challenger.observe_extension_elements::<D>(
        sys,
        addr_coeffs,
        degree >> fri_params.config.rate_bits,
    );
}

pub fn fri_prover_query_rounds(
    sys: &mut System,
    initial_merkle_trees: &Vec<&PolynomialBatch>,
    trees: &Vec<PolynomialBatch>,
    n: usize,
    fri_params: &FriParams,
) {
    let mut rng = rand::thread_rng();
    let random_number: usize = rng.gen();
    for i in 0..fri_params.config.num_query_rounds {
        fri_prover_query_round(
            sys,
            initial_merkle_trees,
            trees,
            random_number % n,
            fri_params,
            i,
        )
    }
}

pub fn fri_prover_query_round(
    sys: &mut System,
    initial_merkle_trees: &Vec<&PolynomialBatch>,
    trees: &Vec<PolynomialBatch>,
    mut x_index: usize,
    fri_params: &FriParams,
    round: usize,
) {
    let mut initial_proof_length = 0;
    let cap_height = fri_params.config.cap_height;
    for tree in initial_merkle_trees {
        initial_proof_length += tree.num_layers(cap_height);
    }
    let initial_proof_id = format!("initial_proof_{}", round);

    let mut mks = Vec::new();

    let addr_initial_proof = sys
        .mem
        .alloc(
            &initial_proof_id,
            initial_proof_length * NUM_HASH_OUT_ELTS * SIZE_F,
        )
        .unwrap();
    initial_proof_length = 0;
    for tree in initial_merkle_trees {
        mks.extend(tree.prove(
            x_index,
            cap_height,
            addr_initial_proof + initial_proof_length * NUM_HASH_OUT_ELTS * SIZE_F,
        ));
        initial_proof_length += tree.num_layers(cap_height);
    }

    let mut merkle_proof_length = 0;
    for tree in trees {
        merkle_proof_length += tree.num_layers(cap_height);
    }
    let merkle_proof_id = format!("merkle_proof_{}", round);
    let addr_merkle_proof = sys
        .mem
        .alloc(
            &merkle_proof_id,
            merkle_proof_length * NUM_HASH_OUT_ELTS * SIZE_F,
        )
        .unwrap();
    merkle_proof_length = 0;
    for (i, tree) in trees.iter().enumerate() {
        let arity_bits = fri_params.reduction_arity_bits[i];
        mks.extend(tree.prove(
            x_index >> arity_bits,
            cap_height,
            addr_merkle_proof + merkle_proof_length * NUM_HASH_OUT_ELTS * SIZE_F,
        ));
        merkle_proof_length += tree.num_layers(cap_height);
        x_index >>= arity_bits;
    }
    sys.run_vec(mks);
}
