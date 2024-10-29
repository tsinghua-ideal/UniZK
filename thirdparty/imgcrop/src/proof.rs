use plonky2::{
    hash::hashing::{SPONGE_CAPACITY, SPONGE_WIDTH},
    plonk::{
        circuit_data::{CircuitData, CommonCircuitData},
        proof::{CompressedProofWithPublicInputs, ProofWithPublicInputs},
    },
    util::serialization::Write,
};
use plonky2_field::types::Field;
use serde::{Deserialize, Serialize};

use crate::{C, D, F};
use anyhow::Result;

pub struct ChunkProof {
    pub proof: ProofWithPublicInputs<F, C, D>,
}

impl ChunkProof {
    pub const ORIGINAL_INIT_STATE_PI_INDEXES: (usize, usize) = (0, SPONGE_WIDTH);
    pub const ORIGINAL_FINAL_STATE_PI_INDEXES: (usize, usize) = (SPONGE_WIDTH, SPONGE_WIDTH * 2);
    pub const EDITED_INIT_STATE_PI_INDEXES: (usize, usize) = (SPONGE_WIDTH * 2, SPONGE_WIDTH * 3);
    pub const EDITED_FINAL_STATE_PI_INDEXES: (usize, usize) = (SPONGE_WIDTH * 3, SPONGE_WIDTH * 4);

    pub fn init_state_public_inputs(&self) -> [F; SPONGE_WIDTH] {
        let mut init_state_public_inputs = [F::ZERO; SPONGE_WIDTH];
        init_state_public_inputs.clone_from_slice(&self.proof.public_inputs[..SPONGE_WIDTH]);
        init_state_public_inputs
    }

    pub fn final_state_public_inputs(&self) -> [F; SPONGE_WIDTH] {
        let mut final_state_public_inputs = [F::ZERO; SPONGE_WIDTH];
        final_state_public_inputs
            .clone_from_slice(&self.proof.public_inputs[SPONGE_WIDTH..SPONGE_WIDTH * 2]);
        final_state_public_inputs
    }
}

#[derive(Serialize, Deserialize)]
pub struct TransformationProof {
    pub(crate) proof: CompressedProofWithPublicInputs<F, C, D>,
}

impl TransformationProof {
    pub fn original_hash(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes
            .write_field_vec(&self.proof.public_inputs[..SPONGE_CAPACITY])
            .unwrap();
        bytes
    }

    pub fn edited_hash(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes
            .write_field_vec(&self.proof.public_inputs[SPONGE_CAPACITY..SPONGE_CAPACITY * 2])
            .unwrap();
        bytes
    }

    pub fn verify(&self, circuit: CircuitData<F, C, D>) -> Result<()> {
        circuit.verify_compressed(self.proof.clone())
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.proof.to_bytes()
    }

    pub fn from_bytes(bytes: Vec<u8>, common_data: &CommonCircuitData<F, D>) -> Result<Self> {
        Ok(TransformationProof {
            proof: CompressedProofWithPublicInputs::from_bytes(bytes, common_data)?,
        })
    }
}
