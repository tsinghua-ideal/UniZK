use std::iter::once;
use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_util::log2_ceil;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use crate::stark::Stark;
use crate::util::from_u32_array_be;

pub mod constants;
pub mod generation;
pub mod layout;

use generation::Sha2TraceGenerator;
use layout::*;

use self::constants::{HASH_IV, ROUND_CONSTANTS};

/// compute field_representation of a sequence of 32 bits interpreted big-endian u32
macro_rules! bit_decomp_32 {
    ($row:expr, $col_fn:ident, $f:ty, $p:ty) => {
        ((0..32).fold(<$p>::ZEROS, |acc, i| {
            acc + $row[$col_fn(i)] * <$f>::from_canonical_u64(1 << i)
        }))
    };
}

macro_rules! bit_decomp_32_ext {
    ($builder:ident, $row:expr, $col_fn:ident, $f:ty) => {
        ((0..32).fold($builder.zero_extension(), |acc, i| {
            let tmp_shift = $builder.constant_extension(<$f>::from_canonical_u64(1 << i));
            let tmp = $builder.mul_extension($row[$col_fn(i)], tmp_shift);
            $builder.add_extension(acc, tmp)
        }))
    };
}
/// Computes the arithmetic generalization of `xor(x, y)`, i.e. `x + y - 2 x y`.
pub(crate) fn xor_gen<P: PackedField>(x: P, y: P) -> P {
    x + y - x * y.doubles()
}

pub(crate) fn xor_gen_ext<F: RichField + Extendable<D>, const D: usize>(
    builder: &mut CircuitBuilder<F, D>,
    x: ExtensionTarget<D>,
    y: ExtensionTarget<D>,
) -> ExtensionTarget<D> {
    let add = builder.add_extension(x, y);
    let two = builder.two_extension();
    let double = builder.mul_extension(y, two);
    let mul = builder.mul_extension(x, double);
    builder.sub_extension(add, mul)
}

#[derive(Copy, Clone)]
pub struct Sha2CompressionStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> Sha2CompressionStark<F, D> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Default for Sha2CompressionStark<F, D> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

const COLUMNS: usize = NUM_COLS;
const PUBLIC_INPUTS: usize = 0;

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for Sha2CompressionStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize> = StarkFrame<P, P::Scalar,  COLUMNS,  PUBLIC_INPUTS>
    where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>;

    type EvaluationFrameTarget =
        StarkFrame<ExtensionTarget<D>, ExtensionTarget<D>, COLUMNS, PUBLIC_INPUTS>;

    fn eval_packed_generic<FE, P, const D2: usize>(
        &self,
        vars: &Self::EvaluationFrame<FE, P, D2>,
        yield_constr: &mut ConstraintConsumer<P>,
    ) where
        FE: FieldExtension<D2, BaseField = F>,
        P: PackedField<Scalar = FE>,
    {
        let curr_row = vars.get_local_values();
        let next_row = vars.get_next_values();

        // set hash idx to 1 at the start. hash_idx should be 1-indexed.
        yield_constr.constraint_first_row(P::ONES - curr_row[HASH_IDX]);

        // set his to initial values at start of hash
        let is_hash_start = curr_row[step_bit(0)];
        for i in 0..8 {
            // degree 3
            yield_constr.constraint(
                is_hash_start * (curr_row[h_i(i)] - FE::from_canonical_u32(HASH_IV[i])),
            );
        }

        // ensure his stay the same outside last two rows of hash
        let his_should_change = next_row[step_bit(64)] + curr_row[step_bit(64)];
        for i in 0..8 {
            // degree 2
            yield_constr.constraint_transition(
                (P::ONES - his_should_change) * (next_row[h_i(i)] - curr_row[h_i(i)]),
            );
        }

        let next_is_phase_0: P = (0..16).map(|i| next_row[step_bit(i)]).sum();
        let next_is_phase_1: P = (16..64).map(|i| next_row[step_bit(i)]).sum();
        let next_is_not_padding = next_is_phase_0 + next_is_phase_1 + next_row[step_bit(64)];

        // increment hash idx if we're at the last step and next isn't padding
        let is_last_step = curr_row[step_bit(64)];
        let transition_to_next_hash = is_last_step * next_is_not_padding;
        // degree 3
        yield_constr.constraint_transition(
            transition_to_next_hash * (P::ONES - (next_row[HASH_IDX] - curr_row[HASH_IDX])),
        );

        // otherwise ensure hash idx stays the same unless next row is padding
        // degree 3
        yield_constr.constraint_transition(
            (P::ONES - is_last_step)
                * next_is_not_padding
                * (next_row[HASH_IDX] - curr_row[HASH_IDX]),
        );

        // load input into wis rotated left by one at start
        // wi
        let decomp = bit_decomp_32!(curr_row, wi_bit, FE, P)
            + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
        yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(0)]));

        // wi minus 2
        let decomp = bit_decomp_32!(curr_row, wi_minus_2_bit, FE, P)
            + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
        yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(14)]));

        // wi minus 15
        let decomp = bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P)
            + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);
        yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(1)]));

        // the rest
        for i in (1..=12).chain(once(14)) {
            let decomp =
                curr_row[wi_field(i)] + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32);

            yield_constr.constraint(is_hash_start * (decomp - curr_row[input_i(i + 1)]));
        }

        // set input filter to 1 iff we're at the start, zero otherwise
        yield_constr.constraint(is_hash_start * (P::ONES - curr_row[INPUT_FILTER]));
        yield_constr.constraint((P::ONES - is_hash_start) * curr_row[INPUT_FILTER]);

        // rotate wis when next step is phase 0 and we're not starting a new hash
        let rotate_wis = next_is_phase_0 * (P::ONES - next_row[step_bit(0)]);
        // shift wis left when next step is phase 1
        let shift_wis = next_is_phase_1;

        // wi next when rotating
        let c = bit_decomp_32!(next_row, wi_bit, FE, P)
            - bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P);
        yield_constr.constraint_transition(rotate_wis * c);

        // wi_minus_2 next
        let decomp = bit_decomp_32!(next_row, wi_minus_2_bit, FE, P);
        yield_constr
            .constraint_transition((rotate_wis + shift_wis) * (decomp - curr_row[wi_field(14)]));

        // wi_minus_15 next
        let decomp = bit_decomp_32!(next_row, wi_minus_15_bit, FE, P);
        yield_constr
            .constraint_transition((rotate_wis + shift_wis) * (decomp - curr_row[wi_field(1)]));

        // wi_minus_1 next
        let decomp = bit_decomp_32!(curr_row, wi_bit, FE, P);
        yield_constr
            .constraint_transition((rotate_wis + shift_wis) * (next_row[wi_field(14)] - decomp));

        // wi_i_minus_3 next
        let decomp = bit_decomp_32!(curr_row, wi_minus_2_bit, FE, P);
        yield_constr
            .constraint_transition((rotate_wis + shift_wis) * (next_row[wi_field(12)] - decomp));

        for i in 1..12 {
            yield_constr.constraint_transition(
                (rotate_wis + shift_wis) * (next_row[wi_field(i)] - curr_row[wi_field(i + 1)]),
            );
        }

        // round fn in phase 0 or 1
        let is_phase_0_or_1: P = (0..64).map(|i| curr_row[step_bit(i)]).sum();

        // S1 := (e >>> 6) xor (e >>> 11) xor (e >>> 25)
        for bit in 0..32 {
            let computed_bit = xor_gen(
                curr_row[e_bit((bit + 6) % 32)],
                curr_row[e_bit((bit + 11) % 32)],
            );
            // degree 3
            yield_constr
                .constraint(is_phase_0_or_1 * (curr_row[xor_tmp_2_bit(bit)] - computed_bit));

            let computed_bit = xor_gen(
                curr_row[xor_tmp_2_bit(bit)],
                curr_row[e_bit((bit + 25) % 32)],
            );
            // degree 3
            yield_constr.constraint(is_phase_0_or_1 * (curr_row[big_s1_bit(bit)] - computed_bit));
        }

        // ch := (e and f) xor ((not e) and g)
        for bit in 0..32 {
            let computed_bit = curr_row[e_bit(bit)] * curr_row[f_bit(bit)];
            // degree 3
            yield_constr.constraint(is_phase_0_or_1 * (curr_row[e_and_f_bit(bit)] - computed_bit));

            let computed_bit = (P::ONES - curr_row[e_bit(bit)]) * curr_row[g_bit(bit)];
            // degree 3
            yield_constr
                .constraint(is_phase_0_or_1 * (curr_row[not_e_and_g_bit(bit)] - computed_bit));

            let computed_bit = xor_gen(curr_row[e_and_f_bit(bit)], curr_row[not_e_and_g_bit(bit)]);
            // degree 3
            yield_constr.constraint(is_phase_0_or_1 * (curr_row[ch_bit(bit)] - computed_bit));
        }

        // S0 := (a >>> 2) xor (a >>> 13) xor (a >>> 22)
        for bit in 0..32 {
            let computed_bit = xor_gen(
                curr_row[a_bit((bit + 2) % 32)],
                curr_row[a_bit((bit + 13) % 32)],
            );
            // degree 3
            yield_constr
                .constraint(is_phase_0_or_1 * (curr_row[xor_tmp_3_bit(bit)] - computed_bit));

            let computed_bit = xor_gen(
                curr_row[xor_tmp_3_bit(bit)],
                curr_row[a_bit((bit + 22) % 32)],
            );
            // degree 3
            yield_constr.constraint(is_phase_0_or_1 * (curr_row[big_s0_bit(bit)] - computed_bit));
        }

        // maj := (a and b) xor (a and c) xor (b and c)
        for bit in 0..32 {
            // degree 3
            yield_constr.constraint(
                is_phase_0_or_1
                    * (curr_row[a_and_b_bit(bit)] - curr_row[a_bit(bit)] * curr_row[b_bit(bit)]),
            );

            // degree 3
            yield_constr.constraint(
                is_phase_0_or_1
                    * (curr_row[a_and_c_bit(bit)] - curr_row[a_bit(bit)] * curr_row[c_bit(bit)]),
            );

            // degree 3
            yield_constr.constraint(
                is_phase_0_or_1
                    * (curr_row[b_and_c_bit(bit)] - curr_row[b_bit(bit)] * curr_row[c_bit(bit)]),
            );

            let computed_bit = xor_gen(curr_row[a_and_b_bit(bit)], curr_row[a_and_c_bit(bit)]);
            // degree 3
            yield_constr
                .constraint(is_phase_0_or_1 * (curr_row[xor_tmp_4_bit(bit)] - computed_bit));

            let computed_bit = xor_gen(curr_row[xor_tmp_4_bit(bit)], curr_row[b_and_c_bit(bit)]);
            // degree 3
            yield_constr.constraint(is_phase_0_or_1 * (curr_row[maj_bit(bit)] - computed_bit));
        }

        // set round constant
        for step in 0..64 {
            // degree 2
            yield_constr.constraint(
                curr_row[step_bit(step)]
                    * (curr_row[KI] - FE::from_canonical_u32(ROUND_CONSTANTS[step])),
            );
        }

        // temp1 := h + S1 + ch + k[i] + w[i]
        // e := d + temp1
        let h_field = curr_row[H_COL];
        let big_s1_field = bit_decomp_32!(curr_row, big_s1_bit, FE, P);
        let ch_field = bit_decomp_32!(curr_row, ch_bit, FE, P);
        let wi_u32 = bit_decomp_32!(curr_row, wi_bit, FE, P);
        let temp1_minus_ki = h_field + big_s1_field + ch_field + wi_u32;

        let d_field = curr_row[D_COL];
        let e_u32_next = bit_decomp_32!(next_row, e_bit, FE, P);

        // degree 2
        yield_constr.constraint(
            is_phase_0_or_1 * (curr_row[E_NEXT_FIELD] - (d_field + temp1_minus_ki + curr_row[KI])),
        );
        // degree 3
        yield_constr.constraint_transition(
            is_phase_0_or_1
                * (curr_row[E_NEXT_FIELD]
                    - (e_u32_next + curr_row[E_NEXT_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
        );

        // temp2 := S0 + maj
        // a := temp1 + temp2
        let s0_field = bit_decomp_32!(curr_row, big_s0_bit, FE, P);
        let maj_field = bit_decomp_32!(curr_row, maj_bit, FE, P);
        let temp2 = s0_field + maj_field;
        let a_u32_next = bit_decomp_32!(next_row, a_bit, FE, P);

        // degree 2
        yield_constr.constraint(
            is_phase_0_or_1 * (curr_row[A_NEXT_FIELD] - (temp2 + temp1_minus_ki + curr_row[KI])),
        );
        // degree 3
        yield_constr.constraint_transition(
            is_phase_0_or_1
                * (curr_row[A_NEXT_FIELD]
                    - (a_u32_next + curr_row[A_NEXT_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
        );

        // update local vars when not in last step
        // h := g
        // g := f
        // f := e
        // d := c
        // c := b
        // b := a
        let decomp = bit_decomp_32!(curr_row, g_bit, FE, P);
        yield_constr.constraint_transition(is_phase_0_or_1 * (next_row[H_COL] - decomp));

        let decomp = bit_decomp_32!(curr_row, c_bit, FE, P);
        yield_constr.constraint_transition(is_phase_0_or_1 * (next_row[D_COL] - decomp));
        for bit in 0..32 {
            // degree 2
            yield_constr.constraint_transition(
                is_phase_0_or_1 * (next_row[g_bit(bit)] - curr_row[f_bit(bit)]),
            );
            yield_constr.constraint_transition(
                is_phase_0_or_1 * (next_row[f_bit(bit)] - curr_row[e_bit(bit)]),
            );
            yield_constr.constraint_transition(
                is_phase_0_or_1 * (next_row[c_bit(bit)] - curr_row[b_bit(bit)]),
            );
            yield_constr.constraint_transition(
                is_phase_0_or_1 * (next_row[b_bit(bit)] - curr_row[a_bit(bit)]),
            );
        }

        // update his in last step of phase 1
        let update_his = curr_row[step_bit(63)];
        let vars = [
            bit_decomp_32!(next_row, a_bit, FE, P),
            bit_decomp_32!(next_row, b_bit, FE, P),
            bit_decomp_32!(next_row, c_bit, FE, P),
            next_row[D_COL],
            bit_decomp_32!(next_row, e_bit, FE, P),
            bit_decomp_32!(next_row, f_bit, FE, P),
            bit_decomp_32!(next_row, g_bit, FE, P),
            next_row[H_COL],
        ];

        for i in 0..8 {
            // degree 2
            yield_constr.constraint_transition(
                update_his * (curr_row[h_i_next_field(i)] - (curr_row[h_i(i)] + vars[i])),
            );

            // degree 3
            yield_constr.constraint_transition(
                update_his
                    * (curr_row[h_i_next_field(i)]
                        - (next_row[h_i(i)]
                            + curr_row[h_i_next_quotient(i)] * FE::from_canonical_u64(1 << 32))),
            );
        }

        // set output to his during last step, 0 otherwise
        for i in 0..8 {
            // degree 2
            yield_constr.constraint(
                is_last_step
                    * (curr_row[output_i(i)]
                        - (curr_row[h_i(i)]
                            + curr_row[HASH_IDX] * FE::from_canonical_u64(1 << 32))),
            );
            yield_constr.constraint((P::ONES - is_last_step) * curr_row[output_i(i)])
        }

        // set output filter to 1 iff it's the last step, zero otherwise
        yield_constr.constraint(is_last_step * (P::ONES - curr_row[OUTPUT_FILTER]));
        yield_constr.constraint((P::ONES - is_last_step) * curr_row[OUTPUT_FILTER]);

        // message schedule to get next row's wi when next row is in phase 1

        let do_msg_schedule = next_is_phase_1;

        // s0 := (w[i-15] >>> 7) xor (w[i-15] >>> 18) xor (w[i-15] >> 3)
        for bit in 0..29 {
            let computed_bit = xor_gen(
                next_row[wi_minus_15_bit((bit + 7) % 32)],
                next_row[wi_minus_15_bit((bit + 18) % 32)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[xor_tmp_0_bit(bit)] - computed_bit),
            );

            let computed_bit = xor_gen(
                next_row[xor_tmp_0_bit(bit)],
                next_row[wi_minus_15_bit(bit + 3)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[little_s0_bit(bit)] - computed_bit),
            );
        }
        for bit in 29..32 {
            // we can ignore the second XOR in this case since it's with 0
            let computed_bit = xor_gen(
                next_row[wi_minus_15_bit((bit + 7) % 32)],
                next_row[wi_minus_15_bit((bit + 18) % 32)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[little_s0_bit(bit)] - computed_bit),
            );
        }

        // s1 := (w[i-2] >>> 17) xor (w[i-2] >>> 19) xor (w[i-2] >> 10)
        for bit in 0..22 {
            let computed_bit = xor_gen(
                next_row[wi_minus_2_bit((bit + 17) % 32)],
                next_row[wi_minus_2_bit((bit + 19) % 32)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[xor_tmp_1_bit(bit)] - computed_bit),
            );

            let computed_bit = xor_gen(
                next_row[xor_tmp_1_bit(bit)],
                next_row[wi_minus_2_bit(bit + 10)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[little_s1_bit(bit)] - computed_bit),
            );
        }
        for bit in 22..32 {
            // we can ignore the second XOR in this case since it's with 0
            let computed_bit = xor_gen(
                next_row[wi_minus_2_bit((bit + 17) % 32)],
                next_row[wi_minus_2_bit((bit + 19) % 32)],
            );
            // degree 3
            yield_constr.constraint_transition(
                do_msg_schedule * (next_row[little_s1_bit(bit)] - computed_bit),
            );
        }

        // w[i] := w[i-16] + s0 + w[i-7] + s1

        // degree 1
        let s0_field_computed = bit_decomp_32!(next_row, little_s0_bit, FE, P);
        let s1_field_computed = bit_decomp_32!(next_row, little_s1_bit, FE, P);
        let wi_minus_16_field = bit_decomp_32!(curr_row, wi_minus_15_bit, FE, P);
        let wi_minus_7_field = next_row[wi_field(8)];
        let wi = bit_decomp_32!(next_row, wi_bit, FE, P);

        // degree 2
        yield_constr.constraint_transition(
            do_msg_schedule
                * (next_row[WI_FIELD]
                    - (wi_minus_16_field
                        + s0_field_computed
                        + wi_minus_7_field
                        + s1_field_computed)),
        );
        // degree 3
        yield_constr.constraint(
            do_msg_schedule
                * (next_row[WI_FIELD]
                    - (wi + next_row[WI_QUOTIENT] * FE::from_canonical_u64(1 << 32))),
        );

        // set initial step bits to a 1 followed by NUM_STEPS_PER_HASH-1 0s
        yield_constr.constraint_first_row(P::ONES - curr_row[step_bit(0)]);
        for step in 1..NUM_STEPS_PER_HASH {
            yield_constr.constraint_first_row(curr_row[step_bit(step)]);
        }

        // inc step bits when next is not padding
        for bit in 0..NUM_STEPS_PER_HASH {
            // degree 3
            yield_constr.constraint_transition(
                next_is_not_padding
                    * (next_row[step_bit((bit + 1) % NUM_STEPS_PER_HASH)]
                        - curr_row[step_bit(bit)]),
            );
        }

        eval_bits_are_bits(curr_row, yield_constr);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let curr_row = vars.get_local_values();
        let next_row = vars.get_next_values();

        // set hash idx to 1 at the start. hash_idx should be 1-indexed.
        let one = builder.one_extension();
        let first_row_constraints = builder.sub_extension(one, curr_row[HASH_IDX]);
        yield_constr.constraint_first_row(builder, first_row_constraints);

        // set his to initial values at start of hash
        let is_hash_start = curr_row[step_bit(0)];
        for i in 0..8 {
            let constant = builder.constant_extension(F::Extension::from_canonical_u32(HASH_IV[i]));
            let tmp = builder.sub_extension(curr_row[h_i(i)], constant);
            let tmp = builder.mul_extension(is_hash_start, tmp);
            yield_constr.constraint(builder, tmp);
        }

        // ensure his stay the same outside last two rows of hash
        let his_should_change =
            builder.add_extension(next_row[step_bit(64)], curr_row[step_bit(64)]);
        for i in 0..8 {
            let tmp = builder.sub_extension(next_row[h_i(i)], curr_row[h_i(i)]);
            let sub = builder.sub_extension(one, his_should_change);
            let tmp = builder.mul_extension(sub, tmp);
            yield_constr.constraint_transition(builder, tmp);
        }

        let next_is_phase_0 = (0..16).fold(builder.zero_extension(), |acc, i| {
            builder.add_extension(acc, next_row[step_bit(i)])
        });
        let next_is_phase_1 = (16..64).fold(builder.zero_extension(), |acc, i| {
            builder.add_extension(acc, next_row[step_bit(i)])
        });
        let next_is_not_padding = {
            let tmp = builder.add_extension(next_is_phase_0, next_is_phase_1);
            builder.add_extension(tmp, next_row[step_bit(64)])
        };

        // increment hash idx if we're at the last step and next isn't padding
        let is_last_step = curr_row[step_bit(64)];
        let transition_to_next_hash = builder.mul_extension(is_last_step, next_is_not_padding);

        let constraint = {
            let tmp = builder.sub_extension(next_row[HASH_IDX], curr_row[HASH_IDX]);
            let sub = builder.sub_extension(one, tmp);
            builder.mul_extension(transition_to_next_hash, sub)
        };
        yield_constr.constraint_transition(builder, constraint);

        // otherwise ensure hash idx stays the same unless next row is padding
        let constraint = {
            let tmp = builder.sub_extension(next_row[HASH_IDX], curr_row[HASH_IDX]);
            let sub = builder.sub_extension(one, is_last_step);
            let mul = builder.mul_extension(sub, next_is_not_padding);
            builder.mul_extension(mul, tmp)
        };
        yield_constr.constraint_transition(builder, constraint);

        // load input into wis rotated left by one at start
        // wi
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let mul = builder.mul_extension(curr_row[HASH_IDX], tmp_shift);
        let decomp = builder.add_extension(decomp, mul);
        let sub = builder.sub_extension(decomp, curr_row[input_i(0)]);
        let constraint = builder.mul_extension(is_hash_start, sub);
        yield_constr.constraint(builder, constraint);

        // wi_minus_2
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_minus_2_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let mul = builder.mul_extension(curr_row[HASH_IDX], tmp_shift);
        let decomp = builder.add_extension(decomp, mul);
        let sub = builder.sub_extension(decomp, curr_row[input_i(14)]);
        let constraint = builder.mul_extension(is_hash_start, sub);
        yield_constr.constraint(builder, constraint);

        // wi_minus_15
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_minus_15_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let mul = builder.mul_extension(curr_row[HASH_IDX], tmp_shift);
        let decomp = builder.add_extension(decomp, mul);
        let sub = builder.sub_extension(decomp, curr_row[input_i(1)]);
        let constraint = builder.mul_extension(is_hash_start, sub);
        yield_constr.constraint(builder, constraint);

        // the rest
        for i in (1..=12).chain(once(14)) {
            let decomp = builder.add_extension(curr_row[wi_field(i)], mul);
            let sub = builder.sub_extension(decomp, curr_row[input_i(i + 1)]);
            let constraint = builder.mul_extension(is_hash_start, sub);
            yield_constr.constraint(builder, constraint);
        }

        // set input filter to 1 iff we're at the start, zero otherwise
        let sub = builder.sub_extension(one, curr_row[INPUT_FILTER]);
        let constraint = builder.mul_extension(is_hash_start, sub);
        yield_constr.constraint(builder, constraint);

        let sub = builder.sub_extension(one, is_hash_start);
        let constraint = builder.mul_extension(sub, curr_row[INPUT_FILTER]);
        yield_constr.constraint(builder, constraint);

        // rotate wis when next step is phase 0 and we're not starting a new hash
        let sub = builder.sub_extension(one, next_row[step_bit(0)]);
        let rotate_wis = builder.mul_extension(next_is_phase_0, sub);
        // shift wis left when next step is phase 1
        let shift_wis = next_is_phase_1;

        // wi next when rotating
        let decomp1 = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(next_row[wi_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let decomp2 = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_minus_15_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let c = builder.sub_extension(decomp1, decomp2);
        let constraint = builder.mul_extension(rotate_wis, c);
        yield_constr.constraint_transition(builder, constraint);

        // wi_minus_2 next
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(next_row[wi_minus_2_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let sub = builder.sub_extension(decomp, curr_row[wi_field(14)]);
        let add = builder.add_extension(rotate_wis, shift_wis);
        let constraint = builder.mul_extension(add, sub);
        yield_constr.constraint_transition(builder, constraint);

        // wi_minus_15 next
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(next_row[wi_minus_15_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let sub = builder.sub_extension(decomp, curr_row[wi_field(1)]);
        let add = builder.add_extension(rotate_wis, shift_wis);
        let constraint = builder.mul_extension(add, sub);
        yield_constr.constraint_transition(builder, constraint);

        // wi_minus_1 next
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let sub = builder.sub_extension(next_row[wi_field(14)], decomp);
        let add = builder.add_extension(rotate_wis, shift_wis);
        let constraint = builder.mul_extension(add, sub);
        yield_constr.constraint_transition(builder, constraint);

        // wi_i_minus_3 next
        let decomp = (0..32).fold(builder.zero_extension(), |acc, i| {
            let tmp_shift = builder.constant_extension(F::Extension::from_canonical_u64(1 << i));
            let tmp = builder.mul_extension(curr_row[wi_minus_2_bit(i)], tmp_shift);
            builder.add_extension(acc, tmp)
        });
        let sub = builder.sub_extension(next_row[wi_field(12)], decomp);
        let add = builder.add_extension(rotate_wis, shift_wis);
        let constraint = builder.mul_extension(add, sub);
        yield_constr.constraint_transition(builder, constraint);

        for i in 1..12 {
            let sub = builder.sub_extension(next_row[wi_field(i)], curr_row[wi_field(i + 1)]);
            let add = builder.add_extension(rotate_wis, shift_wis);
            let constraint = builder.mul_extension(add, sub);
            yield_constr.constraint_transition(builder, constraint);
        }

        // round fn in phase 0 or 1
        let is_phase_0_or_1 = (0..64).fold(builder.zero_extension(), |acc, i| {
            builder.add_extension(acc, curr_row[step_bit(i)])
        });

        // S1 := (e >>> 6) xor (e >>> 11) xor (e >>> 25)
        for bit in 0..32 {
            let computed_bit = xor_gen_ext(
                builder,
                curr_row[e_bit((bit + 6) % 32)],
                curr_row[e_bit((bit + 11) % 32)],
            );
            let sub = builder.sub_extension(curr_row[xor_tmp_2_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);

            let computed_bit = xor_gen_ext(
                builder,
                curr_row[xor_tmp_2_bit(bit)],
                curr_row[e_bit((bit + 25) % 32)],
            );
            let sub = builder.sub_extension(curr_row[big_s1_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);
        }

        // ch := (e and f) xor ((not e) and g)
        for bit in 0..32 {
            let computed_bit = builder.mul_extension(curr_row[e_bit(bit)], curr_row[f_bit(bit)]);
            let sub = builder.sub_extension(curr_row[e_and_f_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);

            let sub = builder.sub_extension(one, curr_row[e_bit(bit)]);
            let computed_bit = builder.mul_extension(sub, curr_row[g_bit(bit)]);
            let sub = builder.sub_extension(curr_row[not_e_and_g_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);

            let computed_bit = xor_gen_ext(
                builder,
                curr_row[e_and_f_bit(bit)],
                curr_row[not_e_and_g_bit(bit)],
            );
            let sub = builder.sub_extension(curr_row[ch_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);
        }

        // S0 := (a >>> 2) xor (a >>> 13) xor (a >>> 22)
        for bit in 0..32 {
            let computed_bit = xor_gen_ext(
                builder,
                curr_row[a_bit((bit + 2) % 32)],
                curr_row[a_bit((bit + 13) % 32)],
            );
            let sub = builder.sub_extension(curr_row[xor_tmp_3_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);

            let computed_bit = xor_gen_ext(
                builder,
                curr_row[xor_tmp_3_bit(bit)],
                curr_row[a_bit((bit + 22) % 32)],
            );
            let sub = builder.sub_extension(curr_row[big_s0_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);
        }

        // maj := (a and b) xor (a and c) xor (b and c)
        for bit in 0..32 {
            let sub = builder.mul_extension(curr_row[a_bit(bit)], curr_row[b_bit(bit)]);
            let constraint = builder.sub_extension(curr_row[a_and_b_bit(bit)], sub);
            let constraint = builder.mul_extension(is_phase_0_or_1, constraint);
            yield_constr.constraint(builder, constraint);

            let sub = builder.mul_extension(curr_row[a_bit(bit)], curr_row[c_bit(bit)]);
            let constraint = builder.sub_extension(curr_row[a_and_c_bit(bit)], sub);
            let constraint = builder.mul_extension(is_phase_0_or_1, constraint);
            yield_constr.constraint(builder, constraint);

            let sub = builder.mul_extension(curr_row[b_bit(bit)], curr_row[c_bit(bit)]);
            let constraint = builder.sub_extension(curr_row[b_and_c_bit(bit)], sub);
            let constraint = builder.mul_extension(is_phase_0_or_1, constraint);
            yield_constr.constraint(builder, constraint);

            let computed_bit = xor_gen_ext(
                builder,
                curr_row[a_and_b_bit(bit)],
                curr_row[a_and_c_bit(bit)],
            );
            let sub = builder.sub_extension(curr_row[xor_tmp_4_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);

            let computed_bit = xor_gen_ext(
                builder,
                curr_row[xor_tmp_4_bit(bit)],
                curr_row[b_and_c_bit(bit)],
            );
            let sub = builder.sub_extension(curr_row[maj_bit(bit)], computed_bit);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint(builder, constraint);
        }

        // set round constant
        for step in 0..64 {
            let constant =
                builder.constant_extension(F::Extension::from_canonical_u32(ROUND_CONSTANTS[step]));
            let sub = builder.sub_extension(curr_row[KI], constant);
            let constraint = builder.mul_extension(curr_row[step_bit(step)], sub);
            yield_constr.constraint(builder, constraint);
        }

        // temp1 := h + S1 + ch + k[i] + w[i]
        // e := d + temp1
        let h_field = curr_row[H_COL];
        let big_s1_field = bit_decomp_32_ext!(builder, curr_row, big_s1_bit, F::Extension);
        let ch_field = bit_decomp_32_ext!(builder, curr_row, ch_bit, F::Extension);
        let wi_u32 = bit_decomp_32_ext!(builder, curr_row, wi_bit, F::Extension);
        let temp1_minus_ki = {
            let tmp = builder.add_extension(h_field, big_s1_field);
            let tmp = builder.add_extension(tmp, ch_field);
            builder.add_extension(tmp, wi_u32)
        };

        let d_field = curr_row[D_COL];
        let e_u32_next = bit_decomp_32_ext!(builder, next_row, e_bit, F::Extension);

        let tmp = builder.add_extension(d_field, temp1_minus_ki);
        let add = builder.add_extension(tmp, curr_row[KI]);
        let sub = builder.sub_extension(curr_row[E_NEXT_FIELD], add);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint(builder, constraint);

        let constant = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let tmp = builder.mul_extension(curr_row[E_NEXT_QUOTIENT], constant);
        let add = builder.add_extension(e_u32_next, tmp);
        let sub = builder.sub_extension(curr_row[E_NEXT_FIELD], add);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint_transition(builder, constraint);

        // temp2 := S0 + maj
        // a := temp1 + temp2
        let s0_field = bit_decomp_32_ext!(builder, curr_row, big_s0_bit, F::Extension);
        let maj_field = bit_decomp_32_ext!(builder, curr_row, maj_bit, F::Extension);
        let temp2 = builder.add_extension(s0_field, maj_field);
        let a_u32_next = bit_decomp_32_ext!(builder, next_row, a_bit, F::Extension);

        let tmp = builder.add_extension(temp1_minus_ki, temp2);
        let add = builder.add_extension(tmp, curr_row[KI]);
        let sub = builder.sub_extension(curr_row[A_NEXT_FIELD], add);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint(builder, constraint);

        let constant = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let tmp = builder.mul_extension(curr_row[A_NEXT_QUOTIENT], constant);
        let add = builder.add_extension(a_u32_next, tmp);
        let sub = builder.sub_extension(curr_row[A_NEXT_FIELD], add);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint_transition(builder, constraint);

        // update local vars when not in last step
        // h := g
        // g := f
        // f := e
        // d := c
        // c := b
        // b := a
        let decomp = bit_decomp_32_ext!(builder, curr_row, g_bit, F::Extension);
        let sub = builder.sub_extension(next_row[H_COL], decomp);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint_transition(builder, constraint);

        let decomp = bit_decomp_32_ext!(builder, curr_row, c_bit, F::Extension);
        let sub = builder.sub_extension(next_row[D_COL], decomp);
        let constraint = builder.mul_extension(is_phase_0_or_1, sub);
        yield_constr.constraint_transition(builder, constraint);

        for bit in 0..32 {
            let sub = builder.sub_extension(next_row[g_bit(bit)], curr_row[f_bit(bit)]);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint_transition(builder, constraint);

            let sub = builder.sub_extension(next_row[f_bit(bit)], curr_row[e_bit(bit)]);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint_transition(builder, constraint);

            let sub = builder.sub_extension(next_row[c_bit(bit)], curr_row[b_bit(bit)]);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint_transition(builder, constraint);

            let sub = builder.sub_extension(next_row[b_bit(bit)], curr_row[a_bit(bit)]);
            let constraint = builder.mul_extension(is_phase_0_or_1, sub);
            yield_constr.constraint_transition(builder, constraint);
        }

        // update his in last step of phase 1
        let update_his = curr_row[step_bit(63)];
        let vars = [
            bit_decomp_32_ext!(builder, next_row, a_bit, F::Extension),
            bit_decomp_32_ext!(builder, next_row, b_bit, F::Extension),
            bit_decomp_32_ext!(builder, next_row, c_bit, F::Extension),
            next_row[D_COL],
            bit_decomp_32_ext!(builder, next_row, e_bit, F::Extension),
            bit_decomp_32_ext!(builder, next_row, f_bit, F::Extension),
            bit_decomp_32_ext!(builder, next_row, g_bit, F::Extension),
            next_row[H_COL],
        ];

        for i in 0..8 {
            let tmp = builder.add_extension(curr_row[h_i(i)], vars[i]);
            let sub = builder.sub_extension(curr_row[h_i_next_field(i)], tmp);
            let mul = builder.mul_extension(update_his, sub);
            yield_constr.constraint_transition(builder, mul);

            let constant = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
            let tmp = builder.mul_extension(curr_row[h_i_next_quotient(i)], constant);
            let add = builder.add_extension(next_row[h_i(i)], tmp);
            let sub = builder.sub_extension(curr_row[h_i_next_field(i)], add);
            let mul = builder.mul_extension(update_his, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        // set output to his during last step, 0 otherwise
        for i in 0..8 {
            let constant = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
            let tmp = builder.mul_extension(curr_row[HASH_IDX], constant);
            let add = builder.add_extension(curr_row[h_i(i)], tmp);
            let sub = builder.sub_extension(curr_row[output_i(i)], add);
            let mul = builder.mul_extension(is_last_step, sub);
            yield_constr.constraint(builder, mul);

            let sub = builder.sub_extension(one, is_last_step);
            let mul = builder.mul_extension(sub, curr_row[output_i(i)]);
            yield_constr.constraint(builder, mul);
        }

        // set output filter to 1 iff it's the last step, zero otherwise
        let sub = builder.sub_extension(one, curr_row[OUTPUT_FILTER]);
        let mul = builder.mul_extension(is_last_step, sub);
        yield_constr.constraint(builder, mul);

        let sub = builder.sub_extension(one, is_last_step);
        let mul = builder.mul_extension(sub, curr_row[OUTPUT_FILTER]);
        yield_constr.constraint(builder, mul);

        // message schedule to get next row's wi when next row is in phase 1

        let do_msg_schedule = next_is_phase_1;

        // s0 := (w[i-15] >>> 7) xor (w[i-15] >>> 18) xor (w[i-15] >> 3)
        for bit in 0..29 {
            let computed_bit = xor_gen_ext(
                builder,
                next_row[wi_minus_15_bit((bit + 7) % 32)],
                next_row[wi_minus_15_bit((bit + 18) % 32)],
            );
            let sub = builder.sub_extension(next_row[xor_tmp_0_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);

            let computed_bit = xor_gen_ext(
                builder,
                next_row[xor_tmp_0_bit(bit)],
                next_row[wi_minus_15_bit(bit + 3)],
            );
            let sub = builder.sub_extension(next_row[little_s0_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        for bit in 29..32 {
            // we can ignore the second XOR in this case since it's with 0
            let computed_bit = xor_gen_ext(
                builder,
                next_row[wi_minus_15_bit((bit + 7) % 32)],
                next_row[wi_minus_15_bit((bit + 18) % 32)],
            );
            let sub = builder.sub_extension(next_row[little_s0_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        // s1 := (w[i-2] >>> 17) xor (w[i-2] >>> 19) xor (w[i-2] >> 10)
        for bit in 0..22 {
            let computed_bit = xor_gen_ext(
                builder,
                next_row[wi_minus_2_bit((bit + 17) % 32)],
                next_row[wi_minus_2_bit((bit + 19) % 32)],
            );
            let sub = builder.sub_extension(next_row[xor_tmp_1_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);

            let computed_bit = xor_gen_ext(
                builder,
                next_row[xor_tmp_1_bit(bit)],
                next_row[wi_minus_2_bit(bit + 10)],
            );
            let sub = builder.sub_extension(next_row[little_s1_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);
        }
        for bit in 22..32 {
            // we can ignore the second XOR in this case since it's with 0
            let computed_bit = xor_gen_ext(
                builder,
                next_row[wi_minus_2_bit((bit + 17) % 32)],
                next_row[wi_minus_2_bit((bit + 19) % 32)],
            );
            let sub = builder.sub_extension(next_row[little_s1_bit(bit)], computed_bit);
            let mul = builder.mul_extension(do_msg_schedule, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        // w[i] := w[i-16] + s0 + w[i-7] + s1

        // degree 1
        let s0_field_computed = bit_decomp_32_ext!(builder, next_row, little_s0_bit, F::Extension);
        let s1_field_computed = bit_decomp_32_ext!(builder, next_row, little_s1_bit, F::Extension);
        let wi_minus_16_field =
            bit_decomp_32_ext!(builder, curr_row, wi_minus_15_bit, F::Extension);
        let wi_minus_7_field = next_row[wi_field(8)];
        let wi = bit_decomp_32_ext!(builder, next_row, wi_bit, F::Extension);

        // degree 2
        let tmp = builder.add_extension(wi_minus_16_field, s0_field_computed);
        let tmp = builder.add_extension(tmp, wi_minus_7_field);
        let tmp = builder.add_extension(tmp, s1_field_computed);
        let sub = builder.sub_extension(next_row[WI_FIELD], tmp);
        let mul = builder.mul_extension(do_msg_schedule, sub);
        yield_constr.constraint_transition(builder, mul);

        // degree 3
        let constant = builder.constant_extension(F::Extension::from_canonical_u64(1 << 32));
        let tmp = builder.mul_extension(next_row[WI_QUOTIENT], constant);
        let tmp = builder.add_extension(wi, tmp);
        let sub = builder.sub_extension(next_row[WI_FIELD], tmp);
        let mul = builder.mul_extension(do_msg_schedule, sub);
        yield_constr.constraint(builder, mul);

        // set initial step bits to a 1 followed by NUM_STEPS_PER_HASH-1 0s
        let sub = builder.sub_extension(one, curr_row[step_bit(0)]);
        yield_constr.constraint_first_row(builder, sub);
        for step in 1..NUM_STEPS_PER_HASH {
            yield_constr.constraint_first_row(builder, curr_row[step_bit(step)]);
        }

        // inc step bits when next is not padding
        for bit in 0..NUM_STEPS_PER_HASH {
            let sub = builder.sub_extension(
                next_row[step_bit((bit + 1) % NUM_STEPS_PER_HASH)],
                curr_row[step_bit(bit)],
            );
            let mul = builder.mul_extension(next_is_not_padding, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        // step bits
        for bit in 0..NUM_STEPS_PER_HASH {
            let sub = builder.sub_extension(one, curr_row[step_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[step_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        for bit in 0..32 {
            // wi
            let sub = builder.sub_extension(one, curr_row[wi_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[wi_bit(bit)]);
            yield_constr.constraint(builder, mul);

            // wi minus 2
            let sub = builder.sub_extension(one, curr_row[wi_minus_2_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[wi_minus_2_bit(bit)]);
            yield_constr.constraint(builder, mul);

            // wi minus 15
            let sub = builder.sub_extension(one, curr_row[wi_minus_15_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[wi_minus_15_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // s0
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[little_s0_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[little_s0_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // s1
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[little_s1_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[little_s1_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // a
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[a_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[a_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // b
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[b_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[b_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // c
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[c_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[c_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // e
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[e_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[e_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // f
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[f_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[f_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // g
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[g_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[g_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // S0
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[big_s0_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[big_s0_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // S1
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[big_s1_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[big_s1_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // (not e) and g
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[not_e_and_g_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[not_e_and_g_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // e and f
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[e_and_f_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[e_and_f_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // ch
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[ch_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[ch_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // a and b
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[a_and_b_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[a_and_b_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // a and c
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[a_and_c_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[a_and_c_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // b and c
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[b_and_c_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[b_and_c_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // maj
        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[maj_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[maj_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        // tmps
        for bit in 0..29 {
            let sub = builder.sub_extension(one, curr_row[xor_tmp_0_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[xor_tmp_0_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        for bit in 0..22 {
            let sub = builder.sub_extension(one, curr_row[xor_tmp_1_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[xor_tmp_1_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }

        for bit in 0..32 {
            let sub = builder.sub_extension(one, curr_row[xor_tmp_2_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[xor_tmp_2_bit(bit)]);
            yield_constr.constraint(builder, mul);

            let sub = builder.sub_extension(one, curr_row[xor_tmp_3_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[xor_tmp_3_bit(bit)]);
            yield_constr.constraint(builder, mul);

            let sub = builder.sub_extension(one, curr_row[xor_tmp_4_bit(bit)]);
            let mul = builder.mul_extension(sub, curr_row[xor_tmp_4_bit(bit)]);
            yield_constr.constraint(builder, mul);
        }
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}

fn eval_bits_are_bits<F, P>(curr_row: &[P], yield_constr: &mut ConstraintConsumer<P>)
where
    F: Field,
    P: PackedField<Scalar = F>,
{
    // step bits
    for bit in 0..NUM_STEPS_PER_HASH {
        yield_constr.constraint((P::ONES - curr_row[step_bit(bit)]) * curr_row[step_bit(bit)]);
    }

    for bit in 0..32 {
        // wi
        yield_constr.constraint((P::ONES - curr_row[wi_bit(bit)]) * curr_row[wi_bit(bit)]);

        // wi minus 2
        yield_constr
            .constraint((P::ONES - curr_row[wi_minus_2_bit(bit)]) * curr_row[wi_minus_2_bit(bit)]);

        // wi minus 15
        yield_constr.constraint(
            (P::ONES - curr_row[wi_minus_15_bit(bit)]) * curr_row[wi_minus_15_bit(bit)],
        );
    }

    // s0
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[little_s0_bit(bit)]) * curr_row[little_s0_bit(bit)]);
    }

    // s1
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[little_s1_bit(bit)]) * curr_row[little_s1_bit(bit)]);
    }

    // a
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[a_bit(bit)]) * curr_row[a_bit(bit)]);
    }

    // b
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[b_bit(bit)]) * curr_row[b_bit(bit)]);
    }

    // c
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[c_bit(bit)]) * curr_row[c_bit(bit)]);
    }

    // e
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[e_bit(bit)]) * curr_row[e_bit(bit)]);
    }

    // f
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[f_bit(bit)]) * curr_row[f_bit(bit)]);
    }

    // g
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[g_bit(bit)]) * curr_row[g_bit(bit)]);
    }

    // S0
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[big_s0_bit(bit)]) * curr_row[big_s0_bit(bit)]);
    }

    // S1
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[big_s1_bit(bit)]) * curr_row[big_s1_bit(bit)]);
    }

    // (not e) and g
    for bit in 0..32 {
        yield_constr.constraint(
            (P::ONES - curr_row[not_e_and_g_bit(bit)]) * curr_row[not_e_and_g_bit(bit)],
        );
    }

    // e and f
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[e_and_f_bit(bit)]) * curr_row[e_and_f_bit(bit)]);
    }

    // ch
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[ch_bit(bit)]) * curr_row[ch_bit(bit)]);
    }

    // a and b
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[a_and_b_bit(bit)]) * curr_row[a_and_b_bit(bit)]);
    }

    // a and c
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[a_and_c_bit(bit)]) * curr_row[a_and_c_bit(bit)]);
    }

    // b and c
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[b_and_c_bit(bit)]) * curr_row[b_and_c_bit(bit)]);
    }

    // maj
    for bit in 0..32 {
        yield_constr.constraint((P::ONES - curr_row[maj_bit(bit)]) * curr_row[maj_bit(bit)]);
    }

    // tmps
    for bit in 0..29 {
        yield_constr
            .constraint((P::ONES - curr_row[xor_tmp_0_bit(bit)]) * curr_row[xor_tmp_0_bit(bit)]);
    }
    for bit in 0..22 {
        yield_constr
            .constraint((P::ONES - curr_row[xor_tmp_1_bit(bit)]) * curr_row[xor_tmp_1_bit(bit)]);
    }
    for bit in 0..32 {
        yield_constr
            .constraint((P::ONES - curr_row[xor_tmp_2_bit(bit)]) * curr_row[xor_tmp_2_bit(bit)]);

        yield_constr
            .constraint((P::ONES - curr_row[xor_tmp_3_bit(bit)]) * curr_row[xor_tmp_3_bit(bit)]);

        yield_constr
            .constraint((P::ONES - curr_row[xor_tmp_4_bit(bit)]) * curr_row[xor_tmp_4_bit(bit)]);
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sha2StarkCompressor {
    inputs: Vec<([u32; 8], [u32; 8])>,
}

impl Sha2StarkCompressor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_instance(&mut self, left: [u32; 8], right: [u32; 8]) {
        self.inputs.push((left, right));
    }

    /// returns the generated trace against which a proof may be generated
    pub fn generate<F: Field>(self) -> Vec<PolynomialValues<F>> {
        let max_rows = 1 << log2_ceil(self.inputs.len() * NUM_STEPS_PER_HASH);
        let mut generator = Sha2TraceGenerator::<F>::new(max_rows);
        for (left, right) in self.inputs.into_iter() {
            let digest = generator.gen_hash(left, right);
            let digest_bytes = from_u32_array_be(digest);
            let digest_hex = digest_bytes
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>();
            println!("digest: {:?}", digest_hex);
        }

        generator.into_polynomial_values()
    }
}

#[cfg(test)]
mod tests {
    // use anyhow::Result;
    // use generation::to_u32_array_be;
    // use plonky2::hash::hash_types::BytesHash;
    // use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
    // use plonky2::util::timing::TimingTree;
    // use plonky2_field::types::Sample;

    // use super::*;
    // use crate::config::StarkConfig;
    // use crate::prover::prove;
    // use crate::stark_testing::test_stark_low_degree;
    // use crate::verifier::verify_stark_proof;

    // #[test]
    // fn test_stark_degree() -> Result<()> {
    //     const D: usize = 2;
    //     type C = PoseidonGoldilocksConfig;
    //     type F = <C as GenericConfig<D>>::F;
    //     type S = Sha2CompressionStark<F, D>;

    //     let stark = S::new();
    //     test_stark_low_degree(stark)
    // }

    // #[test]
    // fn test_stark_circuit() -> Result<()> {
    //     const D: usize = 2;
    //     type C = PoseidonGoldilocksConfig;
    //     type F = <C as GenericConfig<D>>::F;
    //     type S = Sha2CompressionStark<F, D>;

    //     let stark = S::new();

    //     test_stark_circuit_constraints::<F, C, S, D>(stark)
    // }

    // #[test]
    // fn test_single() -> Result<()> {
    //     const D: usize = 2;
    //     type C = PoseidonGoldilocksConfig;
    //     type F = <C as GenericConfig<D>>::F;
    //     type S = Sha2CompressionStark<F, D>;

    //     let mut left_input = [0u32; 8];
    //     let mut right_input = [0u32; 8];
    //     for i in 0..8 {
    //         left_input[i] = i as u32;
    //         right_input[i] = i as u32 + 8;
    //     }

    //     let mut generator = Sha2TraceGenerator::<F>::new(128);
    //     generator.gen_hash(left_input, right_input);

    //     let config = StarkConfig::standard_fast_config();
    //     let stark = S::new();
    //     let trace = generator.into_polynomial_values();
    //     let mut timing = TimingTree::default();
    //     let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing)?;

    //     verify_stark_proof(stark, proof, &config)?;

    //     Ok(())
    // }

    // #[test]
    // fn test_multiple() -> Result<()> {
    //     const D: usize = 2;
    //     type C = PoseidonGoldilocksConfig;
    //     type F = <C as GenericConfig<D>>::F;
    //     type S = Sha2CompressionStark<F, D>;

    //     let mut compressor = Sha2StarkCompressor::new();

    //     let left = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
    //     let right = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
    //     compressor.add_instance(left, right);

    //     let left = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
    //     let right = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
    //     compressor.add_instance(left, right);

    //     let trace = compressor.generate();

    //     let config = StarkConfig::standard_fast_config();
    //     let stark = S::new();
    //     let mut timing = TimingTree::default();
    //     let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing)?;

    //     verify_stark_proof(stark, proof, &config)?;

    //     Ok(())
    // }
}
