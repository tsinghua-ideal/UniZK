use alloc::vec::Vec;

use itertools::Itertools;
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::util::transpose;

/// A helper function to transpose a row-wise trace and put it in the format that `prove` expects.
pub fn trace_rows_to_poly_values<F: Field, const COLUMNS: usize>(
    trace_rows: Vec<[F; COLUMNS]>,
) -> Vec<PolynomialValues<F>> {
    let trace_row_vecs = trace_rows.into_iter().map(|row| row.to_vec()).collect_vec();
    let trace_col_vecs: Vec<Vec<F>> = transpose(&trace_row_vecs);
    trace_col_vecs
        .into_iter()
        .map(|column| PolynomialValues::new(column))
        .collect()
}

pub fn to_u32_array_be<const N: usize>(block: [u8; N * 4]) -> [u32; N] {
    let mut block_u32 = [0; N];
    for (o, chunk) in block_u32.iter_mut().zip(block.chunks_exact(4)) {
        *o = u32::from_be_bytes(chunk.try_into().unwrap());
    }
    block_u32
}

pub fn from_u32_array_be<const N: usize>(block: [u32; N]) -> [u8; N * 4] {
    let mut block_u8 = [0; N * 4];
    for (o, chunk) in block_u8.chunks_mut(4).zip(block.iter()) {
        o.copy_from_slice(&chunk.to_be_bytes());
    }
    block_u8
}

