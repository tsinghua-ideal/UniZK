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
use starky::config::StarkConfig;
use starky::proof::StarkProofWithPublicInputs;
use starky::prover::prove;
use starky::recursive_verifier::{
    add_virtual_stark_proof_with_pis, set_stark_proof_with_pis_target, verify_stark_proof_circuit,
};
use starky::sha256::{Sha2CompressionStark, Sha2StarkCompressor};
use starky::stark::Stark;
use starky::util::to_u32_array_be;
use starky::verifier::verify_stark_proof;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = Sha2CompressionStark<F, D>;

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

fn main() {
    let args: Vec<String> = env::args().collect();
    let num_hashes = args[1].parse::<i32>().unwrap();
    println!(
        "\n============== num hashes {} =======================================",
        num_hashes
    );

    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Info);
    builder.try_init().unwrap();

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

    let mut timing = TimingTree::new("stark", Level::Debug);
    timing.push("prove", Level::Debug);
    timing.push("gen trace", Level::Debug);
    let trace = compressor.generate();
    timing.pop();
    println!("trace len {} width {}", trace[0].len(), trace.len());

    let config = StarkConfig::standard_fast_config();

    debug!("Num Columns: {}", S::COLUMNS);
    let stark = S::new();
    let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing).unwrap();

    timing.pop();

    timing.push("verify", Level::Debug);

    let proof_size = ByteSize(proof.to_bytes().len() as u64);
    println!("Proof size: {} bytes", proof_size);
    verify_stark_proof(stark, proof.clone(), &config).unwrap();

    timing.pop();
    timing.print();
    
    recursive_proof::<F, C, S, C, D>(stark, proof, &config, true).unwrap();
}
