use plonky2::hash::hashing::SPONGE_WIDTH;
use plonky2::plonk::{circuit_builder::CircuitBuilder, circuit_data::CircuitConfig};
use plonky2_field::types::Field;

use crate::circuit::{TransformationChunkCircuit, TransformationCircuit};
use crate::proof::ChunkProof;
use crate::{hash::build_hash_chunk_circuit, transformations::TransformationLogic};
use crate::{C, D, F};

pub struct TransformationCircuitBuilder<const L: usize> {
    transformation: Box<dyn TransformationLogic<L>>,
    original_len: usize,
    total_chunks: usize,
}

impl<const L: usize> TransformationCircuitBuilder<L> {
    pub fn new(original_len: usize, transformation: Box<dyn TransformationLogic<L>>) -> Self {
        Self {
            transformation,
            original_len: original_len / 4,
            total_chunks: ((original_len / 4) + L - 1) / L,
        }
    }

    fn build_chunk_circuit(
        &self,
        config: &CircuitConfig,
        chunk: usize,
    ) -> TransformationChunkCircuit {
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let chunk_len = if chunk == self.total_chunks - 1 {
            self.original_len % L
        } else {
            L
        };
        let orig_hasher_targets = build_hash_chunk_circuit::<F, D, L>(&mut builder, chunk_len);
        let edit_hasher_targets = build_hash_chunk_circuit::<F, D, L>(&mut builder, chunk_len);

        self.transformation.build_chunk_circuit(
            &mut builder,
            &orig_hasher_targets.input
                [..orig_hasher_targets.input.len() - orig_hasher_targets.padding_len],
            &edit_hasher_targets.input
                [..edit_hasher_targets.input.len() - edit_hasher_targets.padding_len],
            chunk,
        );

        let circuit = builder.build::<C>();
        TransformationChunkCircuit {
            original_chunk: orig_hasher_targets,
            edited_chunk: edit_hasher_targets,
            circuit,
        }
    }

    pub fn build_curcuit(&self) -> TransformationCircuit<L> {
        let config = CircuitConfig::standard_recursion_zk_config();
        let mut chunk_circuits = Vec::new();
        let mut builder = CircuitBuilder::<F, D>::new(config.clone());

        let mut last_original_final_state_target = builder.constants(&[F::ZERO; SPONGE_WIDTH]);
        let mut last_edited_final_state_target = builder.constants(&[F::ZERO; SPONGE_WIDTH]);

        let mut pts = Vec::new();
        let mut ids = Vec::new();
        for chunk in 0..self.total_chunks {
            let chunk_circuit = self.build_chunk_circuit(&config, chunk);
            let pt = builder.add_virtual_proof_with_pis::<C>(&chunk_circuit.circuit.common);

            let original_init_state_target = &pt.public_inputs
                [ChunkProof::ORIGINAL_INIT_STATE_PI_INDEXES.0
                    ..ChunkProof::ORIGINAL_INIT_STATE_PI_INDEXES.1];
            for (left, right) in last_original_final_state_target
                .iter()
                .zip(original_init_state_target.iter())
            {
                builder.connect(*left, *right);
            }
            last_original_final_state_target.copy_from_slice(
                &pt.public_inputs[ChunkProof::ORIGINAL_FINAL_STATE_PI_INDEXES.0
                    ..ChunkProof::ORIGINAL_FINAL_STATE_PI_INDEXES.1],
            );

            let edited_init_state_target = &pt.public_inputs
                [ChunkProof::EDITED_INIT_STATE_PI_INDEXES.0
                    ..ChunkProof::EDITED_INIT_STATE_PI_INDEXES.1];
            for (left, right) in last_edited_final_state_target
                .iter()
                .zip(edited_init_state_target.iter())
            {
                builder.connect(*left, *right);
            }
            last_edited_final_state_target.copy_from_slice(
                &pt.public_inputs[ChunkProof::EDITED_FINAL_STATE_PI_INDEXES.0
                    ..ChunkProof::EDITED_FINAL_STATE_PI_INDEXES.1],
            );

            let inner_data = builder.add_virtual_verifier_data(
                chunk_circuit.circuit.common.config.fri_config.cap_height,
            );

            builder.verify_proof::<C>(&pt, &inner_data, &chunk_circuit.circuit.common);

            chunk_circuits.push(chunk_circuit);
            pts.push(pt);
            ids.push(inner_data);
        }

        builder.register_public_inputs(&last_original_final_state_target[..4]);
        builder.register_public_inputs(&last_edited_final_state_target[..4]);
        builder.print_gate_counts(0);

        let circuit = builder.build::<C>();

        TransformationCircuit {
            circuit,
            chunk_circuits,
            pts,
            ids,
        }
    }
}
