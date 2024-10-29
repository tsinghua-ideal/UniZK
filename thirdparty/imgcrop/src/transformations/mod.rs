use plonky2::{iop::target::Target, plonk::circuit_builder::CircuitBuilder};
use plonky2_field::goldilocks_field::GoldilocksField;
use crate::zk_transformations::Transformation;

use self::crop::build_crop_circuit;

pub mod crop;
pub mod util;

pub trait TransformationLogic<const L: usize> {
    fn build_chunk_circuit(
        &self,
        builder: &mut CircuitBuilder<GoldilocksField, 2>,
        original_chunk: &[Target],
        edited_chunk: &[Target],
        chunk_number: usize,
    );
}

impl<const L: usize> TransformationLogic<L> for Transformation {
    fn build_chunk_circuit(
        &self,
        builder: &mut CircuitBuilder<GoldilocksField, 2>,
        original_chunk: &[Target],
        edited_chunk: &[Target],
        chunk_number: usize,
    ) {
        match self {
            Transformation::Crop {
                orig_w,
                x,
                y,
                w,
                h,
                orig_h: _,
            } => {
                build_crop_circuit::<L>(
                    builder,
                    original_chunk,
                    edited_chunk,
                    chunk_number,
                    *orig_w,
                    *x,
                    *y,
                    *w,
                    *h,
                );
            }
        }
    }
}
