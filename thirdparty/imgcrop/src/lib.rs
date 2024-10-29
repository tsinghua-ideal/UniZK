pub mod builder;
pub mod circuit;
pub mod hash;
pub mod proof;
pub mod transformations;
pub mod util;
pub mod zk_transformations;

use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
