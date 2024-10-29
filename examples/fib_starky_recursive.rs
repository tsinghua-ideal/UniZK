use anyhow::{Context, Result};
use env_logger::Env;
use log::info;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use unizk::config::RamConfig;
use unizk::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};
use unizk::memory::memory_allocator::MemAlloc;
use unizk::starky::constraint_consumer::ConstraintConsumer;
use unizk::starky::stark::EvaluationFrame;
use unizk::system::system::System;
use unizk::util::{BATCH_SIZE, SIZE_F};
use starky::config::StarkConfig;
use starky::fibonacci_stark::FibonacciStark;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove as starky_prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
};
use starky::stark::Stark;
use starky::verifier::verify_stark_proof;

use unizk::plonk::prover::prove_with_partition_witness;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = FibonacciStark<F, D>;

fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
    (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
}

fn eval_packed_generic(
    sys: &mut System,
    vars: &EvaluationFrame,
    yield_constr: &ConstraintConsumer,
) -> Vec<VecOpConfig> {
    let mut res = Vec::new();
    let addr_constraint = sys.mem.get_addr("constraint").unwrap();
    let addr_local_values = vars.addr_local_values;
    let addr_next_values = vars.addr_next_values;
    let addr_public_inputs = vars.addr_public_inputs;

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_local_values,
        addr_input_1: addr_public_inputs + S::PI_INDEX_X0 * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    res.extend(yield_constr.constraint_first_row(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_public_inputs + S::PI_INDEX_X1 * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    res.extend(yield_constr.constraint_first_row(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_public_inputs + S::PI_INDEX_RES * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    res.extend(yield_constr.constraint_last_row(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_values,
        addr_input_1: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint_transition(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_local_values,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_constraint,
        addr_input_1: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint(addr_constraint));

    res
}

fn recursive_proof<
    F: RichField + Extendable<D>,
    C: GenericConfig<D, F = F>,
    S: Stark<F, D> + Copy,
    InnerC: GenericConfig<D, F = F>,
    const D: usize,
>(
    stark: S,
    inner_proof: StarkProofWithPublicInputs<F, InnerC, D>,
    inner_config: &StarkConfig,
    print_gate_counts: bool,
) where
    InnerC::Hasher: AlgebraicHasher<F>,
{
    let mut ramsim = RamConfig::new(&format!("{}", "fib_starky_recursive"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(32, 4096);
    let mut sys = System::new(mem, ramsim);

    let circuit_config = CircuitConfig::standard_recursion_zk_config();
    let mut builder = CircuitBuilder::<F, D>::new(circuit_config);
    let mut pw = PartialWitness::new();
    let degree_bits = inner_proof.proof.recover_degree_bits(inner_config);
    let pt = add_virtual_stark_proof_with_pis(&mut builder, stark, inner_config, degree_bits);
    set_stark_proof_with_pis_target(&mut pw, &pt, &inner_proof);

    verify_stark_proof_circuit::<F, InnerC, S, D>(&mut builder, stark, pt, inner_config);

    if print_gate_counts {
        builder.print_gate_counts(0);
    }

    let data = builder.build::<C>();
    let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
    prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);

    info!("Total number of mem reqs: {}", sys.ramsim.op_cnt);
    info!("Total computations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let config = StarkConfig::standard_fast_config();
    let num_rows = 1 << 20;
    let public_inputs = [F::ZERO, F::ONE, fibonacci(num_rows - 1, F::ZERO, F::ONE)];
    let stark = S::new(num_rows);
    let trace = stark.generate_trace(public_inputs[0], public_inputs[1]);
    let proof = starky_prove::<F, C, S, D>(
        stark,
        &config,
        trace,
        &public_inputs,
        &mut TimingTree::default(),
    )
    .unwrap();
    verify_stark_proof(stark, proof.clone(), &config).unwrap();

    recursive_proof::<F, C, S, C, D>(stark, proof, &config, true);
}
