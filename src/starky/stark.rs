use plonky2::field::extension::{Extendable, FieldExtension};
use plonky2::fri::structure::{FriBatchInfo, FriInstanceInfo, FriOracleInfo, FriPolynomialInfo};
use plonky2::hash::hash_types::RichField;
use starky::config::StarkConfig;
use starky::stark::Stark;

pub struct EvaluationFrame {
    pub addr_local_values: usize,
    pub addr_next_values: usize,
    pub values_len: usize,
    pub addr_public_inputs: usize,
    _public_inputs_len: usize,
}

impl EvaluationFrame {
    pub fn from_values(
        addr_local_values: usize,
        addr_next_values: usize,
        values_len: usize,
        addr_public_inputs: usize,
        _public_inputs_len: usize,
    ) -> Self {
        Self {
            addr_local_values,
            addr_next_values,
            values_len,
            addr_public_inputs,
            _public_inputs_len,
        }
    }
}

pub fn fri_instance<F, S, const D: usize>(
    stark: &S,
    zeta: F::Extension,
    g: F,
    config: &StarkConfig,
) -> FriInstanceInfo<F, D>
where
    F: RichField + Extendable<D>,
    S: Stark<F, D>,
{
    let mut oracles = vec![];

    let trace_info = FriPolynomialInfo::from_range(oracles.len(), 0..S::COLUMNS);
    oracles.push(FriOracleInfo {
        num_polys: S::COLUMNS,
        blinding: false,
    });

    let num_quotient_polys = stark.quotient_degree_factor() * config.num_challenges;
    let quotient_info = FriPolynomialInfo::from_range(oracles.len(), 0..num_quotient_polys);
    oracles.push(FriOracleInfo {
        num_polys: num_quotient_polys,
        blinding: false,
    });

    let zeta_batch = FriBatchInfo {
        point: zeta,
        polynomials: [trace_info.clone(), quotient_info].concat(),
    };
    let zeta_next_batch = FriBatchInfo {
        point: zeta.scalar_mul(g),
        polynomials: [trace_info].concat(),
    };
    let batches = vec![zeta_batch, zeta_next_batch];

    FriInstanceInfo { oracles, batches }
}
