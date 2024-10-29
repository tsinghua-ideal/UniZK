use std::env;

use anyhow::{Context, Result};
use bytesize::ByteSize;
use log::{debug, Level, LevelFilter};
use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{AlgebraicHasher, GenericConfig, PoseidonGoldilocksConfig};
use plonky2::util::timing::TimingTree;
use starky::aes128::generation::AesTraceGenerator;
use starky::aes128::AesStark;
use starky::config::StarkConfig;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove;
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
    println!("gates: {}", data.common.gates.len());
    for (_i, gate) in data.common.gates.iter().enumerate() {
        println!("gate: {}", gate.0.id());
    }
    let proof = data.prove(pw)?;

    let proof_size = ByteSize(proof.to_bytes().len() as u64);
    println!("Recursive proof size: {} bytes", proof_size);
    data.verify(proof)
}

fn main() {
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Debug);
    builder.try_init().unwrap();

    let mut gene = AesTraceGenerator::<F>::new(256);
    // let key = [
    //     0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f,
    //     0x3c,
    // ];
    // let plaintext = [
    //     0x32, 0x43, 0xf6, 0xa8, 0x88, 0x5a, 0x30, 0x8d, 0x31, 0x31, 0x98, 0xa2, 0xe0, 0x37, 0x07,
    //     0x34,
    // ];
    let key = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
    let plaintext = [0u8; 16];

    let output = gene.gen_aes(plaintext, key);
    println!("output {:02x?}", output);
    let trace = gene.into_polynomial_values();

    let mut timing = TimingTree::new("stark", Level::Debug);
    timing.push("prove", Level::Debug);
    timing.push("gen trace", Level::Debug);

    timing.pop();
    println!("trace len {} width {}", trace[0].len(), trace.len());

    let config = StarkConfig::standard_fast_config();

    debug!("Num Columns: {}", S::COLUMNS);
    let stark = S::new();
    let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing).unwrap();

    // let mut buffer = Buffer::new(Vec::new());
    // let _ = buffer.write_stark_proof_with_public_inputs(&proof);
    // println!("proof size {}\n", buffer.bytes().len());

    timing.pop();

    timing.push("verify", Level::Debug);

    let proof_size = ByteSize(proof.to_bytes().len() as u64);
    println!("Proof size: {} bytes", proof_size);
    verify_stark_proof(stark, proof.clone(), &config).unwrap();

    timing.pop();
    timing.print();

    recursive_proof::<F, C, S, C, D>(stark, proof, &config, true).unwrap();
}
