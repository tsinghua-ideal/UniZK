use alloc::vec;
use alloc::vec::Vec;
use core::marker::PhantomData;

use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::field::packed::PackedField;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;

use crate::constraint_consumer::{ConstraintConsumer, RecursiveConstraintConsumer};
use crate::evaluation_frame::{StarkEvaluationFrame, StarkFrame};
use crate::permutation::PermutationPair;
use crate::stark::Stark;
use crate::util::trace_rows_to_poly_values;

#[derive(Copy, Clone)]
pub struct FactorialStark<F: RichField + Extendable<D>, const D: usize> {
    num_rows: usize,
    _phantom: PhantomData<F>,
}

impl<F: RichField + Extendable<D>, const D: usize> FactorialStark<F, D> {
    pub const PI_INDEX_RES: usize = 0;

    pub const fn new(num_rows: usize) -> Self {
        Self {
            num_rows,
            _phantom: PhantomData,
        }
    }

    /// Generate the trace using `x0, x1, 0, 1` as initial state values.
    pub fn generate_trace(&self) -> Vec<PolynomialValues<F>> {
        let trace_rows = (0..self.num_rows)
            .scan([F::TWO, F::ONE], |acc, _| {
                let tmp = *acc;
                acc[0] = tmp[0] + F::ONE;
                acc[1] = tmp[0] * tmp[1];
                Some(tmp)
            })
            .collect::<Vec<_>>();
        trace_rows_to_poly_values(trace_rows)
    }
}

const COLUMNS: usize = 2;
const PUBLIC_INPUTS: usize = 1;

impl<F: RichField + Extendable<D>, const D: usize> Stark<F, D> for FactorialStark<F, D> {
    type EvaluationFrame<FE, P, const D2: usize> = StarkFrame<P, P::Scalar, COLUMNS, PUBLIC_INPUTS>
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
        let local_values = vars.get_local_values();
        let next_values = vars.get_next_values();
        let public_inputs = vars.get_public_inputs();

        // Check public inputs.
        yield_constr.constraint_last_row(local_values[1] - public_inputs[Self::PI_INDEX_RES]);

        yield_constr.constraint_transition(next_values[0] - local_values[0] - P::ONES);
        yield_constr.constraint_transition(next_values[1] - local_values[0] * local_values[1]);
    }

    fn eval_ext_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: &Self::EvaluationFrameTarget,
        yield_constr: &mut RecursiveConstraintConsumer<F, D>,
    ) {
        let local_values = vars.get_local_values();
        let next_values = vars.get_next_values();
        let public_inputs = vars.get_public_inputs();
        // Check public inputs.
        let pis_constraints =
            [builder.sub_extension(local_values[1], public_inputs[Self::PI_INDEX_RES])];
        yield_constr.constraint_last_row(builder, pis_constraints[0]);

        let first_col_constraint = {
            let tmp = builder.sub_extension(next_values[0], local_values[0]);
            let one = builder.one_extension();
            builder.sub_extension(tmp, one)
        };
        yield_constr.constraint_transition(builder, first_col_constraint);
        let second_col_constraint = {
            let tmp = builder.mul_extension(local_values[0], local_values[1]);
            builder.sub_extension(next_values[1], tmp)
        };
        yield_constr.constraint_transition(builder, second_col_constraint);
    }

    fn constraint_degree(&self) -> usize {
        3
    }
}
