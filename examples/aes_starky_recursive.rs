use env_logger::Env;
use log::info;
use std::env;

use plonky2::field::extension::Extendable;
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
use starky::aes128::constants::*;
use starky::aes128::generation::AesTraceGenerator;
use starky::aes128::layout::*;
use starky::aes128::AesStark;
use starky::aes128::{key_3_after_rot_start, key_u8_bit};
use starky::config::StarkConfig;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove as starky_prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
};
use starky::stark::Stark;
use starky::util::to_u32_array_be;
use starky::verifier::verify_stark_proof;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = AesStark<F, D>;

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
    let mut ramsim = RamConfig::new(&format!("{}", "aes_starky_recursive"));
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

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let mut gene = AesTraceGenerator::<F>::new(256);
    let key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let plaintext = [0u8; 16];
    let output = gene.gen_aes(plaintext, key);
    println!("aes output: {:?}", output);

    let trace = gene.into_polynomial_values();

    let config = StarkConfig::standard_fast_config();
    let stark = S::new();
    let proof =
        starky_prove::<F, C, S, D>(stark, &config, trace, &[], &mut TimingTree::default()).unwrap();

    verify_stark_proof(stark, proof.clone(), &config).unwrap();
    recursive_proof::<F, C, S, C, D>(stark, proof, &config, true);
}
