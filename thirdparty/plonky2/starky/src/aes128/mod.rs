use std::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::types::Field;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use crate::stark::Stark;

pub mod aes_impl;
pub mod constants;
pub mod generation;
pub mod layout;

use layout::*;

use self::constants::*;

/// compute field_representation of a sequence of 32 bits interpreted big-endian u32
macro_rules! bit_decomp_32 {
    ($row:expr, $col_start:expr, $f:ty, $p:ty) => {
        ((0..32).fold(<$p>::ZEROS, |acc, i| {
            acc + $row[$col_start + i] * <$f>::from_canonical_u64(1 << i)
        }))
    };
}

macro_rules! bit_decomp_8 {
    ($row:expr, $col_start:expr, $f:ty, $p:ty) => {
        ((0..8).fold(<$p>::ZEROS, |acc, i| {
            acc + $row[$col_start + i] * <$f>::from_canonical_u64(1 << i)
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

fn check_gmul<const D: usize, F: RichField + Extendable<D>, FE, P: PackedField, const D2: usize>(
    yield_constr: &mut ConstraintConsumer<P>,
    curr_row: &[P],
    next_row: &[P],
    sub_step: usize,
    is_gmul_substep: P,
    a_start: usize,   // start col of src a
    b_start: usize,   // start col of src b
    tmp_start: usize, // start col of tmp space for gmul
) where
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    assert!(sub_step < 7);
    let c = 0x1b;

    // a * b_i
    if sub_step == 0 {
        let b_i = curr_row[b_start];
        for j in 0..8 {
            let a_j = curr_row[a_start + j];
            let a_j_b_i = a_j * b_i;
            yield_constr.constraint(
                is_gmul_substep * (curr_row[key_gmul_a_bi_bit(tmp_start, j)] - a_j_b_i),
            );
        }
    }
    let b_i = curr_row[b_start + sub_step + 1];
    for j in 0..8 {
        let a_j = curr_row[key_gmul_a_tmp_bit(tmp_start, j)];
        let a_j_b_i = a_j * b_i;
        yield_constr.constraint_transition(
            is_gmul_substep * (next_row[key_gmul_a_bi_bit(tmp_start, j)] - a_j_b_i),
        );
    }

    // p ^= a
    for j in 0..8 {
        if sub_step == 0 {
            let a_j = curr_row[key_gmul_a_bi_bit(tmp_start, j)];
            yield_constr
                .constraint(is_gmul_substep * (a_j - curr_row[key_gmul_p_bit(tmp_start, j)]));
        }

        let a_j = next_row[key_gmul_a_bi_bit(tmp_start, j)];
        let computed_bit = xor_gen(a_j, curr_row[key_gmul_p_bit(tmp_start, j)]);

        yield_constr.constraint_transition(
            is_gmul_substep * (computed_bit - next_row[key_gmul_p_bit(tmp_start, j)]),
        );
    }

    // a shift xor 0x1b
    if sub_step == 0 {
        for j in 1..8 {
            let a_j = curr_row[a_start + j - 1];
            let c_bit = FE::from_canonical_u32((c >> j) & 1);
            let computed_bit = a_j + c_bit - a_j * c_bit.doubles();
            yield_constr.constraint(
                is_gmul_substep
                    * (computed_bit - curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]),
            );
        }

        yield_constr.constraint(
            is_gmul_substep
                * (curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)] - FE::from_canonical_u32(1)),
        );
    }
    for j in 1..8 {
        let a_j = curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)];
        let c_bit = FE::from_canonical_u32((c >> j) & 1);
        let computed_bit = a_j + c_bit - a_j * c_bit.doubles();
        yield_constr.constraint_transition(
            is_gmul_substep * (computed_bit - next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]),
        );
    }
    yield_constr.constraint_transition(
        is_gmul_substep
            * (next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)] - FE::from_canonical_u32(1)),
    );

    // a xor
    if sub_step == 0 {
        let hi_bit_set = curr_row[a_start + 7];
        for j in 0..8 {
            yield_constr.constraint(
                is_gmul_substep
                    * hi_bit_set
                    * (curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]
                        - curr_row[key_gmul_a_tmp_bit(tmp_start, j)]),
            );
        }
        for j in 1..8 {
            yield_constr.constraint(
                is_gmul_substep
                    * (P::ONES - hi_bit_set)
                    * (curr_row[a_start + j - 1] - curr_row[key_gmul_a_tmp_bit(tmp_start, j)]),
            );
        }
        yield_constr.constraint(
            is_gmul_substep * (P::ONES - hi_bit_set) * curr_row[key_gmul_a_tmp_bit(tmp_start, 0)],
        );
    }
    let hi_bit_set = curr_row[key_gmul_a_tmp_bit(tmp_start, 7)];
    for j in 0..8 {
        yield_constr.constraint_transition(
            is_gmul_substep
                * hi_bit_set
                * (next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)]
                    - next_row[key_gmul_a_tmp_bit(tmp_start, j)]),
        );
    }
    for j in 1..8 {
        yield_constr.constraint_transition(
            is_gmul_substep
                * (P::ONES - hi_bit_set)
                * (curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)]
                    - next_row[key_gmul_a_tmp_bit(tmp_start, j)]),
        );
    }
    yield_constr.constraint_transition(
        is_gmul_substep * (P::ONES - hi_bit_set) * next_row[key_gmul_a_tmp_bit(tmp_start, 0)],
    );
}

fn check_gmul_const<
    const D: usize,
    F: RichField + Extendable<D>,
    FE,
    P: PackedField,
    const D2: usize,
>(
    yield_constr: &mut ConstraintConsumer<P>,
    curr_row: &[P],
    next_row: &[P],
    sub_step: usize,
    is_gmul_substep: P,
    // a is constant
    a: u8,
    b_start: usize,   // start col of src b
    tmp_start: usize, // start col of tmp space for gmul
) where
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    assert!(sub_step < 7);
    let mut t = a;
    let mut temp_a = Vec::with_capacity(8);
    for _ in 0..8 {
        temp_a.push(t);
        let hi_bit_set = (t & 0x80) != 0;
        t <<= 1;
        if hi_bit_set {
            t ^= 0x1b;
        }
    }

    // a * b_i
    if sub_step == 0 {
        let b_i = curr_row[b_start];
        for j in 0..8 {
            let a_j = FE::from_canonical_u32(((temp_a[0] >> j) & 1) as u32);
            let a_j_b_i = b_i * a_j;
            yield_constr.constraint(
                is_gmul_substep * (curr_row[state_gmul_const_a_bi_bit(tmp_start, j)] - a_j_b_i),
            );
        }
    }
    let b_i = curr_row[b_start + sub_step + 1];
    for j in 0..8 {
        let a_j = FE::from_canonical_u32(((temp_a[sub_step + 1] >> j) & 1) as u32);
        let a_j_b_i = b_i * a_j;
        yield_constr.constraint_transition(
            is_gmul_substep * (next_row[state_gmul_const_a_bi_bit(tmp_start, j)] - a_j_b_i),
        );
    }

    // p ^= a
    for j in 0..8 {
        if sub_step == 0 {
            let a_j = curr_row[state_gmul_const_a_bi_bit(tmp_start, j)];
            yield_constr.constraint(
                is_gmul_substep * (curr_row[state_gmul_const_p_bit(tmp_start, j)] - a_j),
            );
        }

        let a_j = next_row[state_gmul_const_a_bi_bit(tmp_start, j)];
        let computed_bit = xor_gen(a_j, curr_row[state_gmul_const_p_bit(tmp_start, j)]);

        yield_constr.constraint_transition(
            is_gmul_substep * (computed_bit - next_row[state_gmul_const_p_bit(tmp_start, j)]),
        );
    }
}

fn check_sub<const D: usize, F: RichField + Extendable<D>, FE, P: PackedField, const D2: usize>(
    yield_constr: &mut ConstraintConsumer<P>,
    curr_row: &[P],
    next_row: &[P],
    steps: &[usize],
    a_start: usize,       // start col of src a
    sub_bit_start: usize, // start col of tmp space for sub
    target_next_row: bool,
    target_bit_start: usize,
) where
    FE: FieldExtension<D2, BaseField = F>,
    P: PackedField<Scalar = FE>,
{
    for sub_step in 0..7 {
        let is_gmul_substep: P = steps
            .iter()
            .map(|x| curr_row[step_bit(*x + sub_step)])
            .sum();
        check_gmul(
            yield_constr,
            curr_row,
            next_row,
            sub_step,
            is_gmul_substep,
            a_start,
            sub_bit_start,
            sub_bit_start + 8,
        );
    }

    let is_sub: P = steps.iter().map(|x| curr_row[step_bit(*x + 7)]).sum();
    let offset = [0, 4, 5, 6, 7];
    for i in 0..8 {
        for j in 1..5 {
            let computed_bit = if j == 1 {
                xor_gen(
                    curr_row[sub_bit_start + i],
                    curr_row[sub_bit_start + (i + offset[j]) % 8],
                )
            } else {
                xor_gen(
                    curr_row[sbox_xor_bit(sub_bit_start, j - 2, i)],
                    curr_row[sub_bit_start + (i + offset[j]) % 8],
                )
            };
            yield_constr.constraint(
                is_sub * (computed_bit - curr_row[sbox_xor_bit(sub_bit_start, j - 1, i)]),
            );
        }

        // ^ 99
        let bit99 = FE::from_canonical_u32(BITS99LSB[i] as u32);
        let bit_tmp = curr_row[sbox_xor_bit(sub_bit_start, 3, i)];
        let computed_bit = bit_tmp + bit99 - bit_tmp * bit99.doubles();
        if target_next_row {
            yield_constr
                .constraint_transition(is_sub * (computed_bit - next_row[target_bit_start + i]));
        } else {
            yield_constr.constraint(is_sub * (computed_bit - curr_row[target_bit_start + i]));
        }
    }
}

fn check_gmul_ext<const D: usize, F: RichField + Extendable<D>>(
    builder: &mut CircuitBuilder<F, D>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    curr_row: &[ExtensionTarget<D>],
    next_row: &[ExtensionTarget<D>],
    sub_step: usize,
    is_gmul_substep: ExtensionTarget<D>,
    a_start: usize,   // start col of src a
    b_start: usize,   // start col of src b
    tmp_start: usize, // start col of tmp space for gmul
) {
    assert!(sub_step < 7);
    let one = builder.one_extension();
    let c = 0x1b;

    // a * b_i
    if sub_step == 0 {
        let b_i = curr_row[b_start];
        for j in 0..8 {
            let a_j = curr_row[a_start + j];
            let a_j_b_i = builder.mul_extension(a_j, b_i);
            let sub = builder.sub_extension(curr_row[key_gmul_a_bi_bit(tmp_start, j)], a_j_b_i);
            let mul = builder.mul_extension(is_gmul_substep, sub);
            yield_constr.constraint(builder, mul);
        }
    }
    let b_i = curr_row[b_start + sub_step + 1];
    for j in 0..8 {
        let a_j = curr_row[key_gmul_a_tmp_bit(tmp_start, j)];
        let a_j_b_i = builder.mul_extension(a_j, b_i);
        let sub = builder.sub_extension(next_row[key_gmul_a_bi_bit(tmp_start, j)], a_j_b_i);
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint_transition(builder, mul);
    }

    // p ^= a
    for j in 0..8 {
        if sub_step == 0 {
            let a_j = curr_row[key_gmul_a_bi_bit(tmp_start, j)];
            let sub = builder.sub_extension(a_j, curr_row[key_gmul_p_bit(tmp_start, j)]);
            let mul = builder.mul_extension(is_gmul_substep, sub);
            yield_constr.constraint(builder, mul);
        }
        let a_j = next_row[key_gmul_a_bi_bit(tmp_start, j)];
        let computed_bit = xor_gen_ext(builder, a_j, curr_row[key_gmul_p_bit(tmp_start, j)]);
        let sub = builder.sub_extension(computed_bit, next_row[key_gmul_p_bit(tmp_start, j)]);
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint_transition(builder, mul);
    }

    // a shift xor 0x1b
    if sub_step == 0 {
        for j in 1..8 {
            let a_j = curr_row[a_start + j - 1];
            let c_bit = builder.constant_extension(F::Extension::from_canonical_u32((c >> j) & 1));
            let computed_bit = xor_gen_ext(builder, a_j, c_bit);
            let sub = builder.sub_extension(
                computed_bit,
                curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)],
            );
            let mul = builder.mul_extension(is_gmul_substep, sub);
            yield_constr.constraint(builder, mul);
        }

        let sub = builder.sub_extension(curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)], one);
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint(builder, mul);
    }
    for j in 1..8 {
        let a_j = curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)];
        let c_bit = builder.constant_extension(F::Extension::from_canonical_u32((c >> j) & 1));
        let computed_bit = xor_gen_ext(builder, a_j, c_bit);
        let sub = builder.sub_extension(
            computed_bit,
            next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)],
        );
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint_transition(builder, mul);
    }
    let sub = builder.sub_extension(next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, 0)], one);
    let mul = builder.mul_extension(is_gmul_substep, sub);
    yield_constr.constraint_transition(builder, mul);

    // a xor
    if sub_step == 0 {
        let hi_bit_set = curr_row[a_start + 7];
        let not_hi_bit_set = builder.sub_extension(one, hi_bit_set);
        for j in 0..8 {
            let sub = builder.sub_extension(
                curr_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)],
                curr_row[key_gmul_a_tmp_bit(tmp_start, j)],
            );
            let mul = builder.mul_extension(is_gmul_substep, hi_bit_set);
            let mul = builder.mul_extension(mul, sub);
            yield_constr.constraint(builder, mul);
        }
        for j in 1..8 {
            let sub = builder.sub_extension(
                curr_row[a_start + j - 1],
                curr_row[key_gmul_a_tmp_bit(tmp_start, j)],
            );
            let mul = builder.mul_extension(is_gmul_substep, not_hi_bit_set);
            let mul = builder.mul_extension(mul, sub);
            yield_constr.constraint(builder, mul);
        }
        let mul = builder.mul_extension(is_gmul_substep, not_hi_bit_set);
        let mul = builder.mul_extension(mul, curr_row[key_gmul_a_tmp_bit(tmp_start, 0)]);
        yield_constr.constraint(builder, mul);
    }
    let hi_bit_set = curr_row[key_gmul_a_tmp_bit(tmp_start, 7)];
    let not_hi_bit_set = builder.sub_extension(one, hi_bit_set);
    for j in 0..8 {
        let sub = builder.sub_extension(
            next_row[key_gmul_a_shift_xor_1b_bit(tmp_start, j)],
            next_row[key_gmul_a_tmp_bit(tmp_start, j)],
        );
        let mul = builder.mul_extension(is_gmul_substep, hi_bit_set);
        let mul = builder.mul_extension(mul, sub);
        yield_constr.constraint_transition(builder, mul);
    }
    for j in 1..8 {
        let sub = builder.sub_extension(
            curr_row[key_gmul_a_tmp_bit(tmp_start, j - 1)],
            next_row[key_gmul_a_tmp_bit(tmp_start, j)],
        );
        let mul = builder.mul_extension(is_gmul_substep, not_hi_bit_set);
        let mul = builder.mul_extension(mul, sub);
        yield_constr.constraint_transition(builder, mul);
    }
    let mul = builder.mul_extension(is_gmul_substep, not_hi_bit_set);
    let mul = builder.mul_extension(mul, next_row[key_gmul_a_tmp_bit(tmp_start, 0)]);
    yield_constr.constraint_transition(builder, mul);
}

fn check_gmul_const_ext<const D: usize, F: RichField + Extendable<D>>(
    builder: &mut CircuitBuilder<F, D>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    curr_row: &[ExtensionTarget<D>],
    next_row: &[ExtensionTarget<D>],
    sub_step: usize,
    is_gmul_substep: ExtensionTarget<D>,
    // a is constant
    a: u32,
    b_start: usize,   // start col of src b
    tmp_start: usize, // start col of tmp space for gmul
) {
    assert!(sub_step < 7);
    let mut t = a;
    let mut temp_a = Vec::with_capacity(8);
    for _ in 0..8 {
        temp_a.push(t);
        let hi_bit_set = (t & 0x80) != 0;
        t <<= 1;
        if hi_bit_set {
            t ^= 0x1b;
        }
    }

    // a * b_i
    if sub_step == 0 {
        let b_i = curr_row[b_start];
        for j in 0..8 {
            let a_j =
                builder.constant_extension(F::Extension::from_canonical_u32((temp_a[0] >> j) & 1));
            let a_j_b_i = builder.mul_extension(b_i, a_j);
            let sub =
                builder.sub_extension(curr_row[state_gmul_const_a_bi_bit(tmp_start, j)], a_j_b_i);
            let mul = builder.mul_extension(is_gmul_substep, sub);
            yield_constr.constraint(builder, mul);
        }
    }
    let b_i = curr_row[b_start + sub_step + 1];
    for j in 0..8 {
        let a_j = builder.constant_extension(F::Extension::from_canonical_u32(
            (temp_a[sub_step + 1] >> j) & 1,
        ));
        let a_j_b_i = builder.mul_extension(b_i, a_j);
        let sub = builder.sub_extension(next_row[state_gmul_const_a_bi_bit(tmp_start, j)], a_j_b_i);
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint_transition(builder, mul);
    }

    // p ^= a
    for j in 0..8 {
        if sub_step == 0 {
            let a_j = curr_row[state_gmul_const_a_bi_bit(tmp_start, j)];
            let sub = builder.sub_extension(curr_row[state_gmul_const_p_bit(tmp_start, j)], a_j);
            let mul = builder.mul_extension(is_gmul_substep, sub);
            yield_constr.constraint(builder, mul);
        }

        let a_j = next_row[state_gmul_const_a_bi_bit(tmp_start, j)];
        let computed_bit =
            xor_gen_ext(builder, a_j, curr_row[state_gmul_const_p_bit(tmp_start, j)]);
        let sub =
            builder.sub_extension(computed_bit, next_row[state_gmul_const_p_bit(tmp_start, j)]);
        let mul = builder.mul_extension(is_gmul_substep, sub);
        yield_constr.constraint_transition(builder, mul);
    }
}

fn check_sub_ext<const D: usize, F: RichField + Extendable<D>>(
    builder: &mut CircuitBuilder<F, D>,
    yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    curr_row: &[ExtensionTarget<D>],
    next_row: &[ExtensionTarget<D>],
    steps: &[usize],
    a_start: usize,       // start col of src a
    sub_bit_start: usize, // start col of tmp space for sub
    target_next_row: bool,
    target_bit_start: usize,
) {
    for sub_step in 0..7 {
        let is_gmul_substep = steps
            .iter()
            .map(|x| curr_row[step_bit(*x + sub_step)])
            .fold(builder.zero_extension(), |acc, x| {
                builder.add_extension(acc, x)
            });
        check_gmul_ext(
            builder,
            yield_constr,
            curr_row,
            next_row,
            sub_step,
            is_gmul_substep,
            a_start,
            sub_bit_start,
            sub_bit_start + 8,
        );
    }

    let is_sub = steps
        .iter()
        .map(|x| curr_row[step_bit(*x + 7)])
        .fold(builder.zero_extension(), |acc, x| {
            builder.add_extension(acc, x)
        });
    let offset = [0, 4, 5, 6, 7];
    for i in 0..8 {
        for j in 1..5 {
            let computed_bit = if j == 1 {
                xor_gen_ext(
                    builder,
                    curr_row[sub_bit_start + i],
                    curr_row[sub_bit_start + (i + offset[j]) % 8],
                )
            } else {
                xor_gen_ext(
                    builder,
                    curr_row[sbox_xor_bit(sub_bit_start, j - 2, i)],
                    curr_row[sub_bit_start + (i + offset[j]) % 8],
                )
            };
            let sub = builder.sub_extension(
                computed_bit,
                curr_row[sbox_xor_bit(sub_bit_start, j - 1, i)],
            );
            let mul = builder.mul_extension(is_sub, sub);
            yield_constr.constraint(builder, mul);
        }

        // ^ 99
        let bit99 =
            builder.constant_extension(F::Extension::from_canonical_u32(BITS99LSB[i] as u32));
        let bit_tmp = curr_row[sbox_xor_bit(sub_bit_start, 3, i)];
        let computed_bit = xor_gen_ext(builder, bit_tmp, bit99);
        if target_next_row {
            let sub = builder.sub_extension(computed_bit, next_row[target_bit_start + i]);
            let mul = builder.mul_extension(is_sub, sub);
            yield_constr.constraint_transition(builder, mul);
        } else {
            let sub = builder.sub_extension(computed_bit, curr_row[target_bit_start + i]);
            let mul = builder.mul_extension(is_sub, sub);
            yield_constr.constraint(builder, mul);
        }
    }
}

fn key_u8_start(i: usize) -> usize {
    assert!(i < 16);
    KEY_START + ((i & 0xC) + 3 - (i & 3)) * 8
}

pub fn key_u8_bit(i: usize, j: usize) -> usize {
    key_u8_start(i) + j
}

pub fn key_3_after_rot_start(i: usize) -> usize {
    let i = (i + 1) % 4;
    key_u8_start(3 * 4 + i)
}

#[derive(Copy, Clone)]
pub struct AesStark<F: RichField + Extendable<D>, const D: usize> {
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> AesStark<F, D> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Default for AesStark<F, D> {
    fn default() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

const COLUMNS: usize = NUM_COLS;
const PUBLIC_INPUTS: usize = 0; //todo

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for AesStark<F, D> {
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

        let next_is_not_padding: P = (0..NUM_STEPS - 1).map(|i| curr_row[step_bit(i)]).sum();

        // check step bits
        yield_constr.constraint_first_row(P::ONES - curr_row[step_bit(0)]);
        for step in 1..NUM_STEPS {
            yield_constr.constraint_first_row(curr_row[step_bit(step)]);
        }

        // inc step bits when next is not padding
        for bit in 0..NUM_STEPS {
            // degree 3
            yield_constr.constraint_transition(
                next_is_not_padding
                    * (next_row[step_bit((bit + 1) % NUM_STEPS)] - curr_row[step_bit(bit)]),
            );
        }

        // check RCON
        let mut start_row = 0;
        for i in 0..10 {
            for step in start_row..ROUND_CHANGE_STEPS[i] + 1 {
                for bit in 0..8 {
                    yield_constr.constraint(
                        curr_row[step_bit(step)]
                            * (curr_row[RCON_START + bit]
                                - FE::from_canonical_u32((RCON[i] >> 24 >> bit) & 1)),
                    );
                }
            }
            start_row = ROUND_CHANGE_STEPS[i] + 1;
        }

        // check add round key
        let is_add_round_key: P = (0..11)
            .map(|i| curr_row[step_bit(ADD_ROUND_KEY_STEPS[i])])
            .sum();

        for i in 0..16 {
            for bit in 0..8 {
                let computed_bit =
                    xor_gen(curr_row[input_i_bit(i, bit)], curr_row[key_u8_bit(i, bit)]);
                yield_constr.constraint_transition(
                    is_add_round_key * (computed_bit - next_row[input_i_bit(i, bit)]),
                );
            }
        }

        // check key update
        // key 3 rot
        check_sub(
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(0),
            key_3_0_bit(0),
            false,
            KEY_3_SUB + 24,
        );
        check_sub(
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(1),
            key_3_1_bit(0),
            false,
            KEY_3_SUB + 16,
        );
        check_sub(
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(2),
            key_3_2_bit(0),
            false,
            KEY_3_SUB + 8,
        );
        check_sub(
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(3),
            key_3_3_bit(0),
            false,
            KEY_3_SUB,
        );

        // check ^ rcon
        let is_key_xor_rcon: P = (0..10)
            .map(|i| curr_row[step_bit(KEY_UPDATE_STEPS[i]) + 7])
            .sum();
        for bit in 0..8 {
            let computed_bit = xor_gen(curr_row[KEY_3_SUB + 24 + bit], curr_row[RCON_START + bit]);
            yield_constr
                .constraint(is_key_xor_rcon * (computed_bit - curr_row[key_3_rconv_bit(bit)]));
        }

        // transition key
        for bit in 0..24 {
            let computed_bit = xor_gen(curr_row[KEY_3_SUB + bit], curr_row[key_i_bit(0, bit)]);
            yield_constr.constraint_transition(
                is_key_xor_rcon * (computed_bit - next_row[key_i_bit(0, bit)]),
            );
        }
        for bit in 0..8 {
            let computed_bit = xor_gen(
                curr_row[key_3_rconv_bit(bit)],
                curr_row[key_i_bit(0, 24 + bit)],
            );
            yield_constr.constraint_transition(
                is_key_xor_rcon * (computed_bit - next_row[key_i_bit(0, 24 + bit)]),
            );
        }
        for i in 1..4 {
            for bit in 0..32 {
                let computed_bit =
                    xor_gen(next_row[key_i_bit(i - 1, bit)], curr_row[key_i_bit(i, bit)]);
                yield_constr.constraint_transition(
                    is_key_xor_rcon * (computed_bit - next_row[key_i_bit(i, bit)]),
                );
            }
        }

        // check keep key
        for i in 0..4 {
            for bit in 0..32 {
                yield_constr.constraint_transition(
                    (P::ONES - is_key_xor_rcon)
                        * next_is_not_padding
                        * (next_row[key_i_bit(i, bit)] - curr_row[key_i_bit(i, bit)]),
                );
            }
        }

        // check state sbox
        for i in 0..16 {
            check_sub(
                yield_constr,
                curr_row,
                next_row,
                &STATE_SBOX_STEPS,
                input_i_bit(i, 0),
                state_i_sub_start(i),
                true,
                input_i_bit(i, 0),
            );
        }

        // check state shift rows
        let is_state_shift_rows: P = (0..10)
            .map(|i| curr_row[step_bit(STATE_SHIFT_ROWS_STEPS[i])])
            .sum();
        for i in 0..16 {
            for j in 0..8 {
                yield_constr.constraint_transition(
                    is_state_shift_rows
                        * (next_row[input_i_bit(i, j)] - curr_row[input_i_bit(SHIFT_ROWS[i], j)]),
                );
            }
        }

        // check state mix columns
        for i in 0..4 {
            for j in 0..4 {
                let b_start = input_i_bit(i * 4 + j, 0);
                for sub_step in 0..7 {
                    let is_gmul_substep: P = STATE_MIX_COLUMNS_STEPS
                        .iter()
                        .map(|x| curr_row[step_bit(*x + sub_step)])
                        .sum();
                    check_gmul_const(
                        yield_constr,
                        curr_row,
                        next_row,
                        sub_step,
                        is_gmul_substep,
                        2,
                        b_start,
                        state_i_gmul_2_start(i * 4 + j),
                    );
                    check_gmul_const(
                        yield_constr,
                        curr_row,
                        next_row,
                        sub_step,
                        is_gmul_substep,
                        3,
                        b_start,
                        state_i_gmul_3_start(i * 4 + j),
                    );
                }
            }

            let is_state_mix_columns = STATE_MIX_COLUMNS_STEPS
                .iter()
                .map(|x| curr_row[step_bit(*x + 7)])
                .sum::<P>();

            for bit in 0..8 {
                // s0
                let computed_bit = xor_gen(
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i), bit)],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 1), bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i, 0) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i, 0) + bit],
                    curr_row[input_i_bit(4 * i + 2, bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i, 1) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i, 1) + bit],
                    curr_row[input_i_bit(4 * i + 3, bit)],
                );
                yield_constr.constraint_transition(
                    is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i, bit)]),
                );

                // s1
                let computed_bit = xor_gen(
                    curr_row[input_i_bit(4 * i, bit)],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 1), bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 1, 0) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 1, 0) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 2), bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 1, 1) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 1, 1) + bit],
                    curr_row[input_i_bit(4 * i + 3, bit)],
                );
                yield_constr.constraint_transition(
                    is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 1, bit)]),
                );

                // s2
                let computed_bit = xor_gen(
                    curr_row[input_i_bit(4 * i, bit)],
                    curr_row[input_i_bit(4 * i + 1, bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 2, 0) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 2, 0) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 2), bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 2, 1) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 2, 1) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 3), bit)],
                );
                yield_constr.constraint_transition(
                    is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 2, bit)]),
                );

                // s3
                let computed_bit = xor_gen(
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i), bit)],
                    curr_row[input_i_bit(4 * i + 1, bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 3, 0) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 3, 0) + bit],
                    curr_row[input_i_bit(4 * i + 2, bit)],
                );
                yield_constr.constraint(
                    is_state_mix_columns
                        * (computed_bit - curr_row[state_i_xor_start(4 * i + 3, 1) + bit]),
                );

                let computed_bit = xor_gen(
                    curr_row[state_i_xor_start(4 * i + 3, 1) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 3), bit)],
                );
                yield_constr.constraint_transition(
                    is_state_mix_columns * (computed_bit - next_row[input_i_bit(4 * i + 3, bit)]),
                );
            }
        }

        // eval bits are bits
        for bit in 0..NUM_COLS {
            yield_constr.constraint((P::ONES - curr_row[bit]) * curr_row[bit]);
        }
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let curr_row = vars.get_local_values();
        let next_row = vars.get_next_values();

        let one = builder.one_extension();

        let next_is_not_padding = (0..NUM_STEPS - 1)
            .map(|i| curr_row[step_bit(i)])
            .fold(builder.zero_extension(), |acc, x| {
                builder.add_extension(acc, x)
            });

        // check step bits
        let sub = builder.sub_extension(one, curr_row[step_bit(0)]);
        yield_constr.constraint_first_row(builder, sub);
        for step in 1..NUM_STEPS {
            yield_constr.constraint_first_row(builder, curr_row[step_bit(step)]);
        }

        // inc step bits when next is not padding
        for bit in 0..NUM_STEPS {
            let next_bit = (bit + 1) % NUM_STEPS;
            let sub = builder.sub_extension(next_row[step_bit(next_bit)], curr_row[step_bit(bit)]);
            let mul = builder.mul_extension(next_is_not_padding, sub);
            yield_constr.constraint_transition(builder, mul);
        }

        // check RCON
        let mut start_row = 0;
        for i in 0..10 {
            for step in start_row..ROUND_CHANGE_STEPS[i] + 1 {
                for bit in 0..8 {
                    let c = builder.constant_extension(F::Extension::from_canonical_u32(
                        (RCON[i] >> 24 >> bit) & 1,
                    ));
                    let sub = builder.sub_extension(curr_row[RCON_START + bit], c);
                    let mul = builder.mul_extension(curr_row[step_bit(step)], sub);
                    yield_constr.constraint(builder, mul);
                }
            }
            start_row = ROUND_CHANGE_STEPS[i] + 1;
        }

        // check add round key
        let is_add_round_key = (0..11)
            .map(|i| curr_row[step_bit(ADD_ROUND_KEY_STEPS[i])])
            .fold(builder.zero_extension(), |acc, x| {
                builder.add_extension(acc, x)
            });

        for i in 0..16 {
            for bit in 0..8 {
                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[input_i_bit(i, bit)],
                    curr_row[key_u8_bit(i, bit)],
                );
                let sub = builder.sub_extension(computed_bit, next_row[input_i_bit(i, bit)]);
                let mul = builder.mul_extension(is_add_round_key, sub);
                yield_constr.constraint_transition(builder, mul);
            }
        }

        // check key update
        // key 3 rot
        check_sub_ext(
            builder,
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(0),
            key_3_0_bit(0),
            false,
            KEY_3_SUB + 24,
        );
        check_sub_ext(
            builder,
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(1),
            key_3_1_bit(0),
            false,
            KEY_3_SUB + 16,
        );
        check_sub_ext(
            builder,
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(2),
            key_3_2_bit(0),
            false,
            KEY_3_SUB + 8,
        );
        check_sub_ext(
            builder,
            yield_constr,
            curr_row,
            next_row,
            &KEY_UPDATE_STEPS,
            key_3_after_rot_start(3),
            key_3_3_bit(0),
            false,
            KEY_3_SUB,
        );

        // check ^ rcon
        let is_key_xor_rcon = (0..10)
            .map(|i| curr_row[step_bit(KEY_UPDATE_STEPS[i]) + 7])
            .fold(builder.zero_extension(), |acc, x| {
                builder.add_extension(acc, x)
            });
        for bit in 0..8 {
            let computed_bit = xor_gen_ext(
                builder,
                curr_row[KEY_3_SUB + 24 + bit],
                curr_row[RCON_START + bit],
            );
            let sub = builder.sub_extension(computed_bit, curr_row[key_3_rconv_bit(bit)]);
            let mul = builder.mul_extension(is_key_xor_rcon, sub);
            yield_constr.constraint(builder, mul);
        }

        // transition key
        for bit in 0..24 {
            let computed_bit = xor_gen_ext(
                builder,
                curr_row[KEY_3_SUB + bit],
                curr_row[key_i_bit(0, bit)],
            );
            let sub = builder.sub_extension(computed_bit, next_row[key_i_bit(0, bit)]);
            let mul = builder.mul_extension(is_key_xor_rcon, sub);
            yield_constr.constraint_transition(builder, mul);
        }
        for bit in 0..8 {
            let computed_bit = xor_gen_ext(
                builder,
                curr_row[key_3_rconv_bit(bit)],
                curr_row[key_i_bit(0, 24 + bit)],
            );
            let sub = builder.sub_extension(computed_bit, next_row[key_i_bit(0, 24 + bit)]);
            let mul = builder.mul_extension(is_key_xor_rcon, sub);
            yield_constr.constraint_transition(builder, mul);
        }
        for i in 1..4 {
            for bit in 0..32 {
                let computed_bit = xor_gen_ext(
                    builder,
                    next_row[key_i_bit(i - 1, bit)],
                    curr_row[key_i_bit(i, bit)],
                );
                let sub = builder.sub_extension(computed_bit, next_row[key_i_bit(i, bit)]);
                let mul = builder.mul_extension(is_key_xor_rcon, sub);
                yield_constr.constraint_transition(builder, mul);
            }
        }

        // check keep key
        for i in 0..4 {
            for bit in 0..32 {
                let sub =
                    builder.sub_extension(next_row[key_i_bit(i, bit)], curr_row[key_i_bit(i, bit)]);
                let is_not_key_xor_rcon = builder.sub_extension(one, is_key_xor_rcon);
                let mul = builder.mul_extension(is_not_key_xor_rcon, next_is_not_padding);
                let mul = builder.mul_extension(mul, sub);
                yield_constr.constraint_transition(builder, mul);
            }
        }

        // check state sbox
        for i in 0..16 {
            check_sub_ext(
                builder,
                yield_constr,
                curr_row,
                next_row,
                &STATE_SBOX_STEPS,
                input_i_bit(i, 0),
                state_i_sub_start(i),
                true,
                input_i_bit(i, 0),
            );
        }

        // check state shift rows
        let is_state_shift_rows = (0..10)
            .map(|i| curr_row[step_bit(STATE_SHIFT_ROWS_STEPS[i])])
            .fold(builder.zero_extension(), |acc, x| {
                builder.add_extension(acc, x)
            });
        for i in 0..16 {
            for j in 0..8 {
                let sub = builder.sub_extension(
                    next_row[input_i_bit(i, j)],
                    curr_row[input_i_bit(SHIFT_ROWS[i], j)],
                );
                let mul = builder.mul_extension(is_state_shift_rows, sub);
                yield_constr.constraint_transition(builder, mul);
            }
        }

        // check state mix columns
        for i in 0..4 {
            for j in 0..4 {
                let b_start = input_i_bit(i * 4 + j, 0);
                for sub_step in 0..7 {
                    let is_gmul_substep = STATE_MIX_COLUMNS_STEPS
                        .iter()
                        .map(|x| curr_row[step_bit(*x + sub_step)])
                        .fold(builder.zero_extension(), |acc, x| {
                            builder.add_extension(acc, x)
                        });
                    check_gmul_const_ext(
                        builder,
                        yield_constr,
                        curr_row,
                        next_row,
                        sub_step,
                        is_gmul_substep,
                        2,
                        b_start,
                        state_i_gmul_2_start(i * 4 + j),
                    );
                    check_gmul_const_ext(
                        builder,
                        yield_constr,
                        curr_row,
                        next_row,
                        sub_step,
                        is_gmul_substep,
                        3,
                        b_start,
                        state_i_gmul_3_start(i * 4 + j),
                    );
                }
            }

            let is_state_mix_columns = STATE_MIX_COLUMNS_STEPS
                .iter()
                .map(|x| curr_row[step_bit(*x + 7)])
                .fold(builder.zero_extension(), |acc, x| {
                    builder.add_extension(acc, x)
                });

            for bit in 0..8 {
                // s0
                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i), bit)],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 1), bit)],
                );
                let sub = builder
                    .sub_extension(computed_bit, curr_row[state_i_xor_start(4 * i, 0) + bit]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i, 0) + bit],
                    curr_row[input_i_bit(4 * i + 2, bit)],
                );
                let sub = builder
                    .sub_extension(computed_bit, curr_row[state_i_xor_start(4 * i, 1) + bit]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i, 1) + bit],
                    curr_row[input_i_bit(4 * i + 3, bit)],
                );
                let sub = builder.sub_extension(computed_bit, next_row[input_i_bit(4 * i, bit)]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint_transition(builder, mul);

                // s1
                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[input_i_bit(4 * i, bit)],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 1), bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 1, 0) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 1, 0) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 2), bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 1, 1) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 1, 1) + bit],
                    curr_row[input_i_bit(4 * i + 3, bit)],
                );
                let sub =
                    builder.sub_extension(computed_bit, next_row[input_i_bit(4 * i + 1, bit)]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint_transition(builder, mul);

                // s2
                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[input_i_bit(4 * i, bit)],
                    curr_row[input_i_bit(4 * i + 1, bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 2, 0) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 2, 0) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 2), bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 2, 1) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 2, 1) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i + 3), bit)],
                );
                let sub =
                    builder.sub_extension(computed_bit, next_row[input_i_bit(4 * i + 2, bit)]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint_transition(builder, mul);

                // s3
                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_gmul_const_res_bit(state_i_gmul_3_start(4 * i), bit)],
                    curr_row[input_i_bit(4 * i + 1, bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 3, 0) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 3, 0) + bit],
                    curr_row[input_i_bit(4 * i + 2, bit)],
                );
                let sub = builder.sub_extension(
                    computed_bit,
                    curr_row[state_i_xor_start(4 * i + 3, 1) + bit],
                );
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint(builder, mul);

                let computed_bit = xor_gen_ext(
                    builder,
                    curr_row[state_i_xor_start(4 * i + 3, 1) + bit],
                    curr_row[state_gmul_const_res_bit(state_i_gmul_2_start(4 * i + 3), bit)],
                );
                let sub =
                    builder.sub_extension(computed_bit, next_row[input_i_bit(4 * i + 3, bit)]);
                let mul = builder.mul_extension(is_state_mix_columns, sub);
                yield_constr.constraint_transition(builder, mul);
            }
        }

        // eval bits are bits
        for bit in 0..NUM_COLS {
            let sub = builder.sub_extension(one, curr_row[bit]);
            let mul = builder.mul_extension(sub, curr_row[bit]);
            yield_constr.constraint(builder, mul);
        }
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}

// #[cfg(test)]
// mod tests {
//     use anyhow::Result;
//     use generation::to_u32_array_be;
//     use plonky2::hash::hash_types::BytesHash;
//     use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
//     use plonky2::util::timing::TimingTree;
//     use plonky2_field::types::Sample;

//     use super::*;
//     use crate::config::StarkConfig;
//     use crate::prover::prove;
//     use crate::stark_testing::test_stark_low_degree;
//     use crate::verifier::verify_stark_proof;

//     #[test]
//     fn test_stark_degree() -> Result<()> {
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = AesStark<F, D>;

//         let stark = S::new();
//         test_stark_low_degree(stark)
//     }

//     // #[test]
//     // fn test_stark_circuit() -> Result<()> {
//     //     const D: usize = 2;
//     //     type C = PoseidonGoldilocksConfig;
//     //     type F = <C as GenericConfig<D>>::F;
//     //     type S = AesStark<F, D>;

//     //     let stark = S::new();

//     //     test_stark_circuit_constraints::<F, C, S, D>(stark)
//     // }

//     #[test]
//     fn test_single() -> Result<()> {
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = AesStark<F, D>;

//         let mut left_input = [0u32; 8];
//         let mut right_input = [0u32; 8];
//         for i in 0..8 {
//             left_input[i] = i as u32;
//             right_input[i] = i as u32 + 8;
//         }

//         let mut generator = AesTraceGenerator::<F>::new(128);
//         generator.gen_hash(left_input, right_input);

//         let config = StarkConfig::standard_fast_config();
//         let stark = S::new();
//         let trace = generator.into_polynomial_values();
//         let mut timing = TimingTree::default();
//         let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing)?;

//         verify_stark_proof(stark, proof, &config)?;

//         Ok(())
//     }

//     #[test]
//     fn test_multiple() -> Result<()> {
//         const D: usize = 2;
//         type C = PoseidonGoldilocksConfig;
//         type F = <C as GenericConfig<D>>::F;
//         type S = AesStark<F, D>;

//         let mut compressor = AesStarkCompressor::new();

//         let left = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
//         let right = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
//         compressor.add_instance(left, right);

//         let left = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
//         let right = to_u32_array_be::<8>(BytesHash::<32>::rand().0);
//         compressor.add_instance(left, right);

//         let trace = compressor.generate();

//         let config = StarkConfig::standard_fast_config();
//         let stark = S::new();
//         let mut timing = TimingTree::default();
//         let proof = prove::<F, C, S, D>(stark, &config, trace, &[], &mut timing)?;

//         verify_stark_proof(stark, proof, &config)?;

//         Ok(())
//     }
// }
