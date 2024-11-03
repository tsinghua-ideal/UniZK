use anyhow::{Context, Result};
use bytesize::ByteSize;
use log::info;
use plonky2::field::extension::Extendable;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use plonky2_maybe_rayon::rayon;
use starky::config::StarkConfig;
use starky::fibonacci_stark::FibonacciStark;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
};
use starky::stark::Stark;
use starky::verifier::verify_stark_proof;

fn fibonacci<F: Field>(n: usize, x0: F, x1: F) -> F {
    (0..n).fold((x0, x1), |x, _| (x.1, x.0 + x.1)).1
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
) -> Result<()>
where
    InnerC::Hasher: AlgebraicHasher<F>,
{
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

    let proof = data.prove(pw)?;

    let proof_size = ByteSize(proof.to_bytes().len() as u64);
    println!("Recursive proof size: {} bytes", proof_size.as_u64());
    data.verify(proof)
}

fn fib() -> Result<()> {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type S = FibonacciStark<F, D>;

    let config = StarkConfig::standard_fast_config();
    let num_rows = 1 << 20;
    let public_inputs = [F::ZERO, F::ONE, fibonacci(num_rows - 1, F::ZERO, F::ONE)];
    let stark = S::new(num_rows);
    let trace = stark.generate_trace(public_inputs[0], public_inputs[1]);
    let proof = prove::<F, C, S, D>(
        stark,
        &config,
        trace,
        &public_inputs,
        &mut TimingTree::default(),
    )?;
    let proof_size = ByteSize(proof.to_bytes().len() as u64);
    println!("Proof size: {} bytes", proof_size.as_u64());

    verify_stark_proof(stark, proof.clone(), &config)?;

    recursive_proof::<F, C, S, C, D>(stark, proof, &config, true)
}

fn main() -> Result<()> {
    let num_cpus = num_cpus::get();
    let threads = num_cpus;
    println!("Number of CPUs: {}", num_cpus);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build thread pool.")?
        .install(|| {
            info!(
                "Using {} compute threads on {} cores",
                rayon::current_num_threads(),
                num_cpus
            );
            // Run the benchmark. `options.lookup_type` determines which benchmark to run.
            fib()
        })?;
    Ok(())
}
