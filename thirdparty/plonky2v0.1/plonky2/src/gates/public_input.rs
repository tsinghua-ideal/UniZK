use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use crate::field::extension::Extendable;
use crate::field::packed::PackedField;
use crate::gates::gate::Gate;
use crate::gates::packed_util::PackedEvaluableBase;
use crate::gates::util::StridedConstraintConsumer;
use crate::hash::hash_types::RichField;
use crate::iop::ext_target::ExtensionTarget;
use crate::iop::generator::WitnessGenerator;
use crate::plonk::circuit_builder::CircuitBuilder;
use crate::plonk::vars::{
    EvaluationTargets, EvaluationVars, EvaluationVarsBase, EvaluationVarsBaseBatch,
    EvaluationVarsBasePacked,
};

/// A gate whose first four wires will be equal to a hash of public inputs.
pub struct PublicInputGate;

impl PublicInputGate {
    pub fn wires_public_inputs_hash() -> Range<usize> {
        0..4
    }
}

impl<F: RichField + Extendable<D>, const D: usize> Gate<F, D> for PublicInputGate {
    fn id(&self) -> String {
        "PublicInputGate".into()
    }

    fn export_circom_verification_code(&self) -> String {
        format!(
            "template PublicInputGateLib() {{
  signal input constants[NUM_OPENINGS_CONSTANTS()][2];
  signal input wires[NUM_OPENINGS_WIRES()][2];
  signal input public_input_hash[4];
  signal input constraints[NUM_GATE_CONSTRAINTS()][2];
  signal output out[NUM_GATE_CONSTRAINTS()][2];

  signal filter[2];
  $SET_FILTER;

  signal hashes[4][2];
  for (var i = 0; i < 4; i++) {{
    hashes[i][0] <== public_input_hash[i];
    hashes[i][1] <== 0;
    out[i] <== ConstraintPush()(constraints[i], filter, GlExtSub()(wires[i], hashes[i]));
  }}
  for (var i = 4; i < NUM_GATE_CONSTRAINTS(); i++) {{
    out[i] <== constraints[i];
  }}
}}"
        )
    }
    fn export_solidity_verification_code(&self) -> String {
        format!(
        "library PublicInputGateLib {{
    using GoldilocksExtLib for uint64[2];
    function set_filter(GatesUtilsLib.EvaluationVars memory ev) internal pure {{
        $SET_FILTER;
    }}
    function eval(GatesUtilsLib.EvaluationVars memory ev, uint64[2][$NUM_GATE_CONSTRAINTS] memory constraints) internal pure {{
        for (uint32 i = 0; i < 4; i++) {{
            GatesUtilsLib.push(constraints, ev.filter, i, ev.wires[i].sub(ev.public_input_hash[i]));
        }}
    }}
}}"
        )
    }

    fn eval_unfiltered(&self, vars: EvaluationVars<F, D>) -> Vec<F::Extension> {
        Self::wires_public_inputs_hash()
            .zip(vars.public_inputs_hash.elements)
            .map(|(wire, hash_part)| vars.local_wires[wire] - hash_part.into())
            .collect()
    }

    fn eval_unfiltered_base_one(
        &self,
        _vars: EvaluationVarsBase<F>,
        _yield_constr: StridedConstraintConsumer<F>,
    ) {
        panic!("use eval_unfiltered_base_packed instead");
    }

    fn eval_unfiltered_base_batch(&self, vars_base: EvaluationVarsBaseBatch<F>) -> Vec<F> {
        self.eval_unfiltered_base_batch_packed(vars_base)
    }

    fn eval_unfiltered_circuit(
        &self,
        builder: &mut CircuitBuilder<F, D>,
        vars: EvaluationTargets<D>,
    ) -> Vec<ExtensionTarget<D>> {
        Self::wires_public_inputs_hash()
            .zip(vars.public_inputs_hash.elements)
            .map(|(wire, hash_part)| {
                let hash_part_ext = builder.convert_to_ext(hash_part);
                builder.sub_extension(vars.local_wires[wire], hash_part_ext)
            })
            .collect()
    }

    fn generators(&self, _row: usize, _local_constants: &[F]) -> Vec<Box<dyn WitnessGenerator<F>>> {
        Vec::new()
    }

    fn num_wires(&self) -> usize {
        4
    }

    fn num_constants(&self) -> usize {
        0
    }

    fn degree(&self) -> usize {
        1
    }

    fn num_constraints(&self) -> usize {
        4
    }
}

impl<F: RichField + Extendable<D>, const D: usize> PackedEvaluableBase<F, D> for PublicInputGate {
    fn eval_unfiltered_base_packed<P: PackedField<Scalar = F>>(
        &self,
        vars: EvaluationVarsBasePacked<P>,
        mut yield_constr: StridedConstraintConsumer<P>,
    ) {
        yield_constr.many(
            Self::wires_public_inputs_hash()
                .zip(vars.public_inputs_hash.elements)
                .map(|(wire, hash_part)| vars.local_wires[wire] - hash_part),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::field::goldilocks_field::GoldilocksField;
    use crate::gates::gate_testing::{test_eval_fns, test_low_degree};
    use crate::gates::public_input::PublicInputGate;
    use crate::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

    #[test]
    fn low_degree() {
        test_low_degree::<GoldilocksField, _, 4>(PublicInputGate)
    }

    #[test]
    fn eval_fns() -> anyhow::Result<()> {
        const D: usize = 2;
        type C = PoseidonGoldilocksConfig;
        type F = <C as GenericConfig<D>>::F;
        test_eval_fns::<F, C, _, D>(PublicInputGate)
    }
}
