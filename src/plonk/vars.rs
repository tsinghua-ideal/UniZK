use crate::util::SIZE_F;

#[derive(Debug, Copy, Clone)]
pub struct EvaluationVarsBaseBatch {
    pub batch_size: usize,
    pub addr_local_constants: usize,
    pub addr_local_wires: usize,
    pub addr_public_inputs_hash: usize,
}
impl EvaluationVarsBaseBatch {
    pub fn new(
        batch_size: usize,
        addr_local_constants: usize,
        addr_local_wires: usize,
        addr_public_inputs_hash: usize,
    ) -> Self {
        Self {
            batch_size,
            addr_local_constants,
            addr_local_wires,
            addr_public_inputs_hash,
        }
    }

    pub fn len(&self) -> usize {
        self.batch_size
    }

    pub fn _view(&self, index: usize) -> EvaluationVarsBaseBatch {
        EvaluationVarsBaseBatch {
            batch_size: self.batch_size,
            addr_local_constants: self.addr_local_constants + index * SIZE_F,
            addr_local_wires: self.addr_local_wires + index * SIZE_F,
            addr_public_inputs_hash: self.addr_public_inputs_hash,
        }
    }
}
