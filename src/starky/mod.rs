use plonky2::util::log2_strict;

use crate::kernel::fft::{Fft, FftConfig, FftDirection};
use crate::system::system::System;
use crate::util::SIZE_F;

pub mod prover;
pub mod constraint_consumer;
pub mod stark;
pub mod vanishing_poly;
pub mod proof;

pub fn lde_onto_coset(
    sys: &mut System,
    addr_input: usize,
    addr_output: usize,
    degree: usize,
    rate_bits: usize,
) {
    let addr_coeffs = sys
        .mem
        .alloc("lde_onto_coset_coeffs", (degree << rate_bits) * SIZE_F)
        .unwrap();

    let ifft_kernel = Fft::new(FftConfig {
        lg_n: log2_strict(degree),
        k: 1,
        direction: FftDirection::NN,
        addr_input: addr_input,
        addr_tmp: 1 << 60,
        addr_output: addr_coeffs,
        inverse: true,
        rate_bits: 0,
        coset: false,
        extension: 1,
        transposed_input: false,
    });
    sys.run_once(&ifft_kernel);

    let fft_kernel = Fft::new(FftConfig {
        lg_n: log2_strict(degree << rate_bits),
        k: 1,
        direction: FftDirection::NN,
        addr_input: addr_coeffs,
        addr_tmp: 1 << 60,
        addr_output: addr_output,
        inverse: false,
        rate_bits: rate_bits,
        coset: true,
        extension: 1,
        transposed_input: false,
    });
    sys.run_once(&fft_kernel);

    sys.mem.free("lde_onto_coset_coeffs");
}
