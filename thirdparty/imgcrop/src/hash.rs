use plonky2::hash::hashing::{PlonkyPermutation, SPONGE_RATE, SPONGE_WIDTH};
use plonky2::iop::target::Target;

use plonky2::hash::hash_types::RichField;
use plonky2::hash::poseidon::{PoseidonHash, PoseidonPermutation};
use plonky2::iop::witness::PartialWitness;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_field::extension::Extendable;

use crate::util::set_multiple_targets;

pub struct ChunkHashTargets {
    init_state: [Target; SPONGE_WIDTH],
    final_state: [Target; SPONGE_WIDTH],
    pub input: Vec<Target>,
    pub padding_len: usize,
}

pub fn build_hash_chunk_circuit<F: RichField + Extendable<D>, const D: usize, const L: usize>(
    builder: &mut CircuitBuilder<F, D>,
    chunk_len: usize,
) -> ChunkHashTargets {
    let (input_len, padding_len) = if chunk_len < L {
        (chunk_len, get_padding_length(chunk_len))
    } else {
        (L, 0)
    };

    let input_targets = builder.add_virtual_targets(input_len + padding_len);
    let init_state_targets = builder.add_virtual_target_arr::<SPONGE_WIDTH>();
    let final_state_targets =
        builder.permute_many::<PoseidonHash>(init_state_targets, &input_targets);

    builder.register_public_inputs(&init_state_targets);
    builder.register_public_inputs(&final_state_targets);

    ChunkHashTargets {
        init_state: init_state_targets,
        final_state: final_state_targets,
        input: input_targets,
        padding_len: padding_len,
    }
}

fn get_padding_length(len: usize) -> usize {
    let padded_length = len + 2;
    ((padded_length + SPONGE_WIDTH - 1) / SPONGE_WIDTH) * SPONGE_WIDTH - len
}

pub struct ChunkHasher<F: RichField + Extendable<D>, const D: usize, const L: usize> {
    data: Vec<F>,
    states: Vec<[F; SPONGE_WIDTH]>,
    current_chunk: usize,
    total_chunks: usize,
}

impl<'a, F: RichField + Extendable<D>, const D: usize, const L: usize> ChunkHasher<F, D, L> {
    pub fn new(data: &[F]) -> Self {
        let mut states = Vec::new();
        let initial_state = [F::ZERO; SPONGE_WIDTH];
        states.push(initial_state);

        Self {
            data: data.to_vec(),
            states: states,
            current_chunk: 0,
            total_chunks: (data.len() + L - 1) / L,
        }
    }

    pub fn total_chunks(&self) -> usize {
        self.total_chunks
    }

    fn pad_chunk(chunk: &mut Vec<F>) {
        chunk.push(F::ONE);
        while (chunk.len() + 1) % SPONGE_WIDTH != 0 {
            chunk.push(F::ZERO);
        }
        chunk.push(F::ONE);
    }

    pub fn will_pad(&self) -> Option<usize> {
        if self.current_chunk == self.total_chunks - 1 {
            Some(self.get_current_chunk().len())
        } else {
            None
        }
    }

    // TO-DO: clean
    fn get_current_chunk(&self) -> &[F] {
        &self.data
            [L * self.current_chunk..std::cmp::min(self.data.len(), L * (self.current_chunk + 1))]
    }

    fn prepare_chunk_prove_data(&self) -> ([F; SPONGE_WIDTH], Vec<F>, [F; SPONGE_WIDTH]) {
        let initial_state = *self.states.last().unwrap();
        let mut state = initial_state.clone();

        let mut chunk = self.get_current_chunk().to_vec();

        // Apply padding for the last chunk
        if self.current_chunk == self.total_chunks - 1 {
            Self::pad_chunk(&mut chunk);
        }

        // Calculate next state
        for input_chunk in chunk.chunks(SPONGE_RATE) {
            state[..input_chunk.len()].copy_from_slice(input_chunk);
            state = PoseidonPermutation::permute(state);
        }

        (initial_state, chunk, state)
    }

    pub fn populate_chunk_inputs(
        &mut self,
        targets: &ChunkHashTargets,
        inputs: &mut PartialWitness<F>,
    ) {
        let (initial_state, chunk, final_state) = self.prepare_chunk_prove_data();
        self.current_chunk += 1;
        self.states.push(final_state);

        set_multiple_targets(inputs, &targets.init_state, &initial_state);
        set_multiple_targets(inputs, &targets.input, &chunk);
        set_multiple_targets(inputs, &targets.final_state, &final_state);
    }
}
