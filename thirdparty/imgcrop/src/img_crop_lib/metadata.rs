use serde::{Deserialize, Serialize};
use crate::zk_transformations::Transformation;
use crate::proof::TransformationProof;

#[derive(Serialize, Deserialize)]
pub struct ProofMetadata {
    pub proof: TransformationProof,
    pub original_length: usize,
    pub edited_length: usize,
    pub transformation: Transformation,
}
