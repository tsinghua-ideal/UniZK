use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::fft::{Fft, FftConfig, FftDirection};
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpConfig, VecOpExtension, VecOpSrc, VecOpType};
use crate::plonk::challenger::Challenger;
use crate::plonk::oracle::PolynomialBatch;
use crate::plonk::zero_poly_coset::ZeroPolyOnCoset;
use crate::starky::constraint_consumer::ConstraintConsumer;
use crate::starky::lde_onto_coset;
use crate::starky::proof::StarkOpeningSet;
use crate::starky::stark::{fri_instance, EvaluationFrame};
use crate::starky::vanishing_poly::eval_vanishing_poly;
use crate::system::system::System;
use crate::util::{BATCH_SIZE, SIZE_F, SPONGE_RATE};
use plonky2::field::extension::Extendable;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::config::GenericConfig;
use plonky2::util::{log2_ceil, log2_strict};
use starky::config::StarkConfig;
use starky::stark::Stark;

pub fn prove<F, C, S, const D: usize>(
    sys: &mut System,
    stark: S,
    config: &StarkConfig,
    trace_poly_values: Vec<PolynomialValues<F>>,
    public_inputs: &[F],
    eval_packed_generic: fn(&mut System, &EvaluationFrame, &ConstraintConsumer) -> Vec<VecOpConfig>,
) where
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D>,
{
    let num_polys = trace_poly_values.len();
    let degree = trace_poly_values[0].len();
    let degree_bits = log2_strict(degree);
    let fri_params = config.fri_params(degree_bits);
    let rate_bits = config.fri_config.rate_bits;
    let cap_height = config.fri_config.cap_height;
    assert!(
        fri_params.total_arities() <= degree_bits + rate_bits - cap_height,
        "FRI total reduction arity is too large.",
    );
    sys.mem
        .alloc("public_inputs", public_inputs.len() * SIZE_F)
        .unwrap();

    let addr_trace = sys
        .mem
        .alloc("trace_cpu", num_polys * degree * SIZE_F)
        .unwrap();

    let trace_commitment = PolynomialBatch::new(
        "trace_commitment",
        sys,
        addr_trace,
        degree,
        num_polys,
        rate_bits,
        false,
        cap_height,
        false,
        Vec::new(),
    );

    let addr_chanllenger_output_buffer = sys
        .mem
        .alloc("Challenger.output_buffer", SPONGE_RATE * SIZE_F)
        .unwrap();
    let mut challenger = Challenger::new(addr_chanllenger_output_buffer);
    challenger.observe_cap(sys, trace_commitment.addr_cap, cap_height);

    let addr_alphas = sys
        .mem
        .alloc("alphas", config.num_challenges * SIZE_F)
        .unwrap();
    challenger.get_n_challenges(sys, addr_alphas, config.num_challenges);

    let quotient_degree_bits = log2_ceil(stark.quotient_degree_factor());
    let addr_quotient_polys = sys
        .mem
        .alloc(
            "quotient_polys",
            config.num_challenges * (degree << quotient_degree_bits) * SIZE_F,
        )
        .unwrap();
    compute_quotient_polys(
        sys,
        &stark,
        &trace_commitment,
        public_inputs,
        addr_alphas,
        degree_bits,
        config,
        eval_packed_generic,
        addr_quotient_polys,
    );

    let quotient_commitment = PolynomialBatch::new(
        "quotient_commitment",
        sys,
        addr_quotient_polys,
        degree,
        config.num_challenges << quotient_degree_bits,
        0,
        false,
        cap_height,
        false,
        Vec::new(),
    );
    sys.mem.free("quotient_polys");

    challenger.observe_cap(
        sys,
        quotient_commitment.addr_cap,
        config.fri_config.cap_height,
    );

    let addr_zeta = sys.mem.alloc("zeta", D * SIZE_F).unwrap();
    challenger.get_n_challenges(sys, addr_zeta, D);

    let addr_g = sys.mem.alloc("g", SIZE_F).unwrap();
    let openings = StarkOpeningSet::<D>::new(
        sys,
        addr_zeta,
        addr_g,
        &trace_commitment,
        &quotient_commitment,
    );

    let addr_zeta_g = sys.mem.alloc("zeta_g", D * SIZE_F).unwrap();
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

    challenger.observe_stark_openings(sys, &openings);
    let initial_merkle_trees = vec![&trace_commitment, &quotient_commitment];
    let instance = fri_instance(&stark, F::Extension::ONE, F::ONE, config);
    PolynomialBatch::prove_openings(
        sys,
        &instance,
        &initial_merkle_trees,
        &mut challenger,
        &fri_params,
        addr_zeta,
        addr_zeta_g,
    );
}

fn compute_quotient_polys<F, S, const D: usize>(
    sys: &mut System,
    stark: &S,
    trace_commitment: &PolynomialBatch,
    public_inputs: &[F],
    addr_alphas: usize,
    degree_bits: usize,
    config: &StarkConfig,
    eval_packed_generic: fn(&mut System, &EvaluationFrame, &ConstraintConsumer) -> Vec<VecOpConfig>,
    addr_res: usize,
) where
    F: RichField + Extendable<D>,
    S: Stark<F, D>,
{
    let degree = 1 << degree_bits;
    let rate_bits = config.fri_config.rate_bits;

    let quotient_degree_bits = log2_ceil(stark.quotient_degree_factor());
    assert!(
        quotient_degree_bits <= rate_bits,
        "Having constraints of degree higher than the rate is not supported yet."
    );
    let step = 1 << (rate_bits - quotient_degree_bits);
    // When opening the `Z`s polys at the "next" point, need to look at the point `next_step` steps away.
    let next_step = 1 << quotient_degree_bits;

    // Evaluation of the first Lagrange polynomial on the LDE domain.
    let addr_lagrange_first_selector = sys
        .mem
        .alloc("lagrange_first_selector", degree * SIZE_F)
        .unwrap();
    let addr_lagrange_first = sys
        .mem
        .alloc("lagrange_first", (degree << quotient_degree_bits) * SIZE_F)
        .unwrap();
    lde_onto_coset(
        sys,
        addr_lagrange_first_selector,
        addr_lagrange_first,
        degree,
        quotient_degree_bits,
    );

    // Evaluation of the last Lagrange polynomial on the LDE domain.
    let addr_lagrange_last_selector = sys
        .mem
        .alloc("lagrange_last_selector", degree * SIZE_F)
        .unwrap();
    let addr_lagrange_last = sys
        .mem
        .alloc("lagrange_last", (degree << quotient_degree_bits) * SIZE_F)
        .unwrap();
    lde_onto_coset(
        sys,
        addr_lagrange_last_selector,
        addr_lagrange_last,
        degree,
        quotient_degree_bits,
    );

    let z_h_on_coset = ZeroPolyOnCoset::new(sys, quotient_degree_bits);
    sys.mem.preload(
        z_h_on_coset.addr_inverses,
        z_h_on_coset.rate.max(BATCH_SIZE),
    );

    // Last element of the subgroup.
    let addr_last = sys.mem.alloc("last", SIZE_F).unwrap();
    let size = degree << quotient_degree_bits;
    let addr_coset = sys.mem.alloc("coset", size * SIZE_F).unwrap();

    let addr_public_inputs = sys.mem.get_addr("public_inputs").unwrap();
    sys.mem.preload(addr_public_inputs, public_inputs.len());
    sys.mem.preload(addr_alphas, config.num_challenges);

    let addr_quotient_values = sys
        .mem
        .alloc("quotient_values", config.num_challenges * size * SIZE_F)
        .unwrap();

    for i_start in (0..size).step_by(BATCH_SIZE) {
        let mut vec_ops = Vec::new();
        let i_next_start = (i_start + next_step) % size;
        let i_range = i_start..i_start + BATCH_SIZE;

        let addr_x = addr_coset + i_range.start * SIZE_F;
        let addr_z_last = sys.mem.alloc("z_last", BATCH_SIZE * SIZE_F).unwrap();
        vec_ops.push(VecOpConfig {
            vector_length: BATCH_SIZE,
            addr_input_0: addr_x,
            addr_input_1: addr_last,
            addr_output: addr_z_last,
            is_final_output: false,
            op_type: VecOpType::SUB,
            op_src: VecOpSrc::VV,
        });

        let mut mks = Vec::new();
        let addr_lagrange_basis_first = addr_lagrange_first + i_range.start * SIZE_F;
        let addr_lagrange_basis_last = addr_lagrange_last + i_range.start * SIZE_F;
        mks.push(MemCpy::new(
            MemCpyConfig {
                addr_input: addr_lagrange_basis_first,
                addr_output: addr_lagrange_basis_first,
                input_length: BATCH_SIZE,
            },
            unsafe { ENABLE_CONFIG.poly },
        ));
        mks.push(MemCpy::new(
            MemCpyConfig {
                addr_input: addr_lagrange_basis_last,
                addr_output: addr_lagrange_basis_last,
                input_length: BATCH_SIZE,
            },
            unsafe { ENABLE_CONFIG.poly },
        ));
        sys.mem.preload(addr_lagrange_basis_first, BATCH_SIZE);
        sys.mem.preload(addr_lagrange_basis_last, BATCH_SIZE);
        sys.mem.preload(addr_z_last, BATCH_SIZE);

        let addr_constraint_accs = (0..config.num_challenges)
            .map(|i| addr_quotient_values + i * size * SIZE_F + i_start * SIZE_F)
            .collect::<Vec<_>>();
        for addr_acc in addr_constraint_accs.iter() {
            sys.mem.preload(*addr_acc, BATCH_SIZE);
        }

        let consumer = ConstraintConsumer::new(
            config.num_challenges,
            addr_alphas,
            addr_z_last,
            addr_lagrange_basis_first,
            addr_lagrange_basis_last,
            addr_constraint_accs,
        );

        let lv = trace_commitment.get_lde_values_packed(i_start, step);
        let nv = trace_commitment.get_lde_values_packed(i_next_start, step);
        let values_len = lv[0].1;

        let addr_lv = sys
            .mem
            .alloc("local_values", BATCH_SIZE * lv[0].1 * SIZE_F)
            .unwrap();
        let addr_nv = sys
            .mem
            .alloc("next_values", BATCH_SIZE * nv[0].1 * SIZE_F)
            .unwrap();
        let mut get_trace_values_packed = |v: Vec<(usize, usize)>, addr_v: usize| {
            for (i, (addr, len)) in v.iter().enumerate() {
                let addr_v_i = addr_v + i * len * SIZE_F;
                let config = MemCpyConfig {
                    addr_input: *addr,
                    addr_output: addr_v_i,
                    input_length: *len,
                };
                let mk = MemCpy::new(config, unsafe { ENABLE_CONFIG.poly });
                mks.push(mk);
            }
        };

        let preload_values = BATCH_SIZE * lv[0].1 <= unsafe { ARCH_CONFIG.num_elems() / 4 };
        if preload_values {
            sys.mem.preload(addr_lv, BATCH_SIZE * lv[0].1);
            sys.mem.preload(addr_nv, BATCH_SIZE * nv[0].1);
        }
        get_trace_values_packed(lv, addr_lv);
        get_trace_values_packed(nv, addr_nv);

        let vars = EvaluationFrame::from_values(
            addr_lv,
            addr_nv,
            values_len,
            addr_public_inputs,
            public_inputs.len(),
        );

        let addr_constraint = sys.mem.alloc("constraint", BATCH_SIZE * SIZE_F).unwrap();
        sys.mem.preload(addr_constraint, BATCH_SIZE);

        vec_ops.extend(eval_vanishing_poly(
            sys,
            &vars,
            &consumer,
            eval_packed_generic,
        ));

        let addr_constraints_evals = consumer.accumulators();
        let addr_denominator_inv = z_h_on_coset.eval_inverse(i_start);
        for i in 0..config.num_challenges {
            let addr_eval = addr_constraints_evals[i];
            vec_ops.push(VecOpConfig {
                vector_length: BATCH_SIZE,
                addr_input_0: addr_eval,
                addr_input_1: addr_denominator_inv,
                addr_output: addr_eval,
                is_final_output: true,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VV,
            });
        }
        sys.run_vec(mks);
        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));

        sys.mem.unpreload(addr_lagrange_basis_first);
        sys.mem.unpreload(addr_lagrange_basis_last);
        sys.mem.unpreload(addr_z_last);
        for addr_acc in addr_constraints_evals.iter() {
            sys.mem.unpreload(*addr_acc);
        }
        if preload_values {
            sys.mem.unpreload(addr_lv);
            sys.mem.unpreload(addr_nv);
        }
        sys.mem.unpreload(addr_constraint);

        sys.mem.free("z_last");
        sys.mem.free("local_values");
        sys.mem.free("next_values");
        sys.mem.free("constraint");
    }
    sys.mem.clear_preload();

    let coset_ifft = Fft::new(FftConfig {
        lg_n: log2_strict(size),
        k: config.num_challenges,
        direction: FftDirection::NN,
        addr_input: addr_quotient_values,
        addr_tmp: 1 << 60,
        addr_output: addr_res,
        inverse: true,
        rate_bits: 0,
        coset: true,
        extension: 1,
        transposed_input: false,
    });
    sys.run_once(&coset_ifft);
}
