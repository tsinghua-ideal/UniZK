use crate::kernel::vector_operation::VecOpConfig;
use crate::starky::constraint_consumer::ConstraintConsumer;
use crate::starky::stark::EvaluationFrame;
use crate::system::system::System;

pub fn eval_vanishing_poly(
    sys: &mut System,
    vars: &EvaluationFrame,
    consumer: &ConstraintConsumer,
    eval_packed_generic: fn(&mut System, &EvaluationFrame, &ConstraintConsumer) -> Vec<VecOpConfig>,
) -> Vec<VecOpConfig> {
    eval_packed_generic(sys, vars, consumer)
}
