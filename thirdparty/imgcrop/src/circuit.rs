use std::time::Instant;

use anyhow::Result;
use itertools::izip;
use log::Level;
use plonky2::{
    hash::hash_types::RichField,
    iop::witness::{PartialWitness, WitnessWrite},
    plonk::{
        circuit_data::{CircuitData, ProverOnlyCircuitData, VerifierCircuitTarget},
        config::{GenericConfig, PoseidonGoldilocksConfig},
        proof::ProofWithPublicInputsTarget,
        prover::prove,
    },
    util::timing::TimingTree,
};
use plonky2_field::{extension::Extendable, goldilocks_field::GoldilocksField};

use crate::{
    hash::{ChunkHashTargets, ChunkHasher},
    proof::{ChunkProof, TransformationProof},
    util::bytes_to_field64,
    C, D, F,
};

pub(crate) struct TransformationChunkCircuit {
    pub(crate) circuit: CircuitData<F, C, D>,
    pub(crate) original_chunk: ChunkHashTargets,
    pub(crate) edited_chunk: ChunkHashTargets,
}

pub struct TransformationCircuit<const L: usize> {
    pub circuit: CircuitData<F, C, D>,
    pub(crate) chunk_circuits: Vec<TransformationChunkCircuit>,
    pub(crate) pts: Vec<ProofWithPublicInputsTarget<D>>,
    pub(crate) ids: Vec<VerifierCircuitTarget>,
}

impl<const L: usize> TransformationCircuit<L> {
    fn prove_chunk(
        &self,
        orig_hasher: &mut ChunkHasher<F, D, L>,
        edit_hasher: &mut ChunkHasher<F, D, L>,
        chunk_curcuit: &TransformationChunkCircuit,
    ) -> Result<ChunkProof> {
        let mut inputs = PartialWitness::<F>::new();
        orig_hasher.populate_chunk_inputs(&chunk_curcuit.original_chunk, &mut inputs);
        edit_hasher.populate_chunk_inputs(&chunk_curcuit.edited_chunk, &mut inputs);

        let mut timing = TimingTree::new("prove_chunk", Level::Info);
        let proof = prove(
            &chunk_curcuit.circuit.prover_only,
            &chunk_curcuit.circuit.common,
            inputs,
            &mut timing,
        )?;
        timing.print();

        chunk_curcuit.circuit.verify(proof.clone())?;

        Ok(ChunkProof { proof })
    }

    fn get_prove_chunk_circuit(
        &self,
        v: &mut Vec<(
            CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>,
            PartialWitness<GoldilocksField>,
        )>,
        chunk_curcuit: &TransformationChunkCircuit,
    ) {
        let inputs = PartialWitness::<F>::new();

        v.push((circuit_clone(&chunk_curcuit.circuit), inputs));
    }

    pub fn get_circuit(
        &mut self,
        original: &[u8],
        edited: &[u8],
    ) -> Vec<(
        CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>,
        PartialWitness<GoldilocksField>,
    )> {
        let mut res = Vec::new();

        let original_elements = bytes_to_field64::<F>(original);
        let edited_elements = bytes_to_field64::<F>(edited);

        println!(
            "Going to proof the hash of {} bytes. {} kB",
            original.len(),
            original.len() / 1024
        );

        let mut orig_hasher = ChunkHasher::<F, D, L>::new(&original_elements);
        let mut edit_hasher = ChunkHasher::<F, D, L>::new(&edited_elements);

        for (chunk_circuit, pt, inner_data) in izip!(&self.chunk_circuits, &self.pts, &self.ids) {
            // println!("Proving chunk...");
            self.get_prove_chunk_circuit(&mut res, chunk_circuit);
        }

        let mut pw = PartialWitness::new();
        for (chunk_circuit, pt, inner_data) in izip!(&self.chunk_circuits, &self.pts, &self.ids) {
            println!("Proving chunk...");
            let chunk_proof = self
                .prove_chunk(&mut orig_hasher, &mut edit_hasher, chunk_circuit)
                .unwrap();
            pw.set_proof_with_pis_target(&pt, &chunk_proof.proof);
            pw.set_verifier_data_target(&inner_data, &chunk_circuit.circuit.verifier_only);
        }

        // res.push((circuit_clone(&self.circuit), pw));

        res
    }

    pub fn prove(&mut self, original: &[u8], edited: &[u8]) -> Result<TransformationProof> {
        let original_elements = bytes_to_field64::<F>(original);
        let edited_elements = bytes_to_field64::<F>(edited);

        println!(
            "Going to proof the hash of {} bytes. {} kB",
            original.len(),
            original.len() / 1024
        );
        let start = Instant::now();

        let mut orig_hasher = ChunkHasher::<F, D, L>::new(&original_elements);
        let mut edit_hasher = ChunkHasher::<F, D, L>::new(&edited_elements);

        let mut pw = PartialWitness::new();
        for (chunk_circuit, pt, inner_data) in izip!(&self.chunk_circuits, &self.pts, &self.ids) {
            println!("Proving chunk...");
            let chunk_proof =
                self.prove_chunk(&mut orig_hasher, &mut edit_hasher, chunk_circuit)?;
            pw.set_proof_with_pis_target(&pt, &chunk_proof.proof);
            pw.set_verifier_data_target(&inner_data, &chunk_circuit.circuit.verifier_only);
        }

        let mut timing = TimingTree::new("prove", Level::Debug);
        let proof = prove(
            &self.circuit.prover_only,
            &self.circuit.common,
            pw,
            &mut timing,
        )?;
        timing.print();

        let duration = start.elapsed();
        println!("Total time for prove is: {:?}", duration);

        Ok(TransformationProof {
            proof: proof.compress(
                &self.circuit.verifier_only.circuit_digest,
                &self.circuit.common,
            )?,
        })
    }
}

fn circuit_clone<F: RichField + Extendable<D>, C: GenericConfig<D, F = F>, const D: usize>(
    c: &CircuitData<F, C, D>,
) -> CircuitData<F, C, D> {
    let new_prover_only = ProverOnlyCircuitData::<F, C, D> {
        generators: Vec::new(),
        generator_indices_by_watches: c.prover_only.generator_indices_by_watches.clone(),
        constants_sigmas_commitment: c.prover_only.constants_sigmas_commitment.clone(),
        sigmas: c.prover_only.sigmas.clone(),
        subgroup: c.prover_only.subgroup.clone(),
        public_inputs: c.prover_only.public_inputs.clone(),
        representative_map: c.prover_only.representative_map.clone(),
        fft_root_table: c.prover_only.fft_root_table.clone(),
        circuit_digest: c.prover_only.circuit_digest.clone(),
    };
    let new_c = CircuitData::<F, C, D> {
        prover_only: new_prover_only,
        verifier_only: c.verifier_only.clone(),
        common: c.common.clone(),
    };
    new_c
}
