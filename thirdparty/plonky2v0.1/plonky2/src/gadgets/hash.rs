use crate::field::extension::Extendable;
use crate::hash::hash_types::{HashOutTarget, RichField};
use crate::hash::hashing::SPONGE_WIDTH;
use crate::iop::target::{BoolTarget, Target};
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::config::AlgebraicHasher;
use crate::hash::hashing::SPONGE_RATE;

impl<F: RichField + Extendable<D>, const D: usize> CircuitBuilder<F, D> {
    pub fn permute<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: [Target; SPONGE_WIDTH],
    ) -> [Target; SPONGE_WIDTH] {
        // We don't want to swap any inputs, so set that wire to 0.
        let _false = self._false();
        self.permute_swapped::<H>(inputs, _false)
    }

    pub fn permute_many<H: AlgebraicHasher<F>>(
        &mut self,
        state: [Target; SPONGE_WIDTH],
        inputs: &[Target],
    ) -> [Target; SPONGE_WIDTH] {
        let mut _state = state.clone();
        for input_chunk in inputs.chunks(SPONGE_RATE) {
            _state[..input_chunk.len()].copy_from_slice(input_chunk);
            _state = self.permute::<H>(_state);
        }
        _state
    }

    /// Conditionally swap two chunks of the inputs (useful in verifying Merkle proofs), then apply
    /// a cryptographic permutation.
    pub(crate) fn permute_swapped<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: [Target; SPONGE_WIDTH],
        swap: BoolTarget,
    ) -> [Target; SPONGE_WIDTH] {
        H::permute_swapped(inputs, swap, self)
    }

    pub fn public_inputs_hash<H: AlgebraicHasher<F>>(
        &mut self,
        inputs: Vec<Target>,
    ) -> HashOutTarget {
        H::public_inputs_hash(inputs, self)
    }
}
