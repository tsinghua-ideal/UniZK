use plonky2::{
    hash::poseidon::PoseidonHash,
    iop::{
        target::Target,
        witness::{PartialWitness, WitnessWrite},
    },
    plonk::config::{GenericHashOut, Hasher},
    util::serialization::Write,
};
use plonky2_field::{
    goldilocks_field::GoldilocksField,
    types::{Field, PrimeField64},
};

pub(crate) fn set_multiple_targets<F: Field>(
    inputs: &mut PartialWitness<F>,
    targets: &[Target],
    values: &[F],
) {
    for (target, value) in targets.iter().zip(values.iter()) {
        inputs.set_target(*target, *value);
    }
}

pub(crate) fn bytes_to_field64<F: PrimeField64>(bytes: &[u8]) -> Vec<F> {
    let mut field_elements = Vec::new();
    for chunk in bytes.chunks(4) {
        let mut elem_bytes = [0u8; 8];
        elem_bytes[..4].copy_from_slice(chunk);
        field_elements.push(F::from_canonical_u64(u64::from_le_bytes(elem_bytes)));
    }
    field_elements
}

#[allow(dead_code)]
pub(crate) fn field64_to_bytes<F: PrimeField64>(field_elements: &[F]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for element in field_elements {
        bytes.write_u32(element.to_canonical_u64() as u32).unwrap()
    }
    bytes
}

pub fn calculate_poseidon(data: &[u8]) -> Vec<u8> {
    PoseidonHash::hash_pad(&bytes_to_field64::<GoldilocksField>(data)).to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialization() {
        let data: Vec<u8> = (u8::MIN..=u8::MAX).collect();
        assert_eq!(
            data,
            field64_to_bytes::<GoldilocksField>(&bytes_to_field64::<GoldilocksField>(&data))
        );
    }
}
