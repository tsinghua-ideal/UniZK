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
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;
use starky::config::StarkConfig;
use starky::factorial_stark::FactorialStark;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove as starky_prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
};
use starky::stark::Stark;
use starky::verifier::verify_stark_proof;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = FactorialStark<F, D>;

fn factorial<F: Field>(n: usize) -> F {
    (2..n + 1).fold(F::ONE, |x, i| x * F::from_canonical_u64(i as u64))
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
    let mut ramsim = RamConfig::new(&format!("{}", "fac_starky_recursive"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
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
    let public_inputs = [factorial(num_rows)];
    let stark = S::new(num_rows);
    let trace = stark.generate_trace();
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
