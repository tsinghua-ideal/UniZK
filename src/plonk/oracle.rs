use log::info;
use plonky2::field::extension::Extendable;
use plonky2::fri::structure::{FriBatchInfo, FriInstanceInfo};
use plonky2::fri::FriParams;
use plonky2::hash::hash_types::RichField;
use plonky2::util::log2_strict;

use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::fft::{Fft, FftConfig, FftDirection};
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::tree::{Tree, TreeConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpConfig, VecOpExtension, VecOpSrc, VecOpType};
use crate::plonk::challenger::Challenger;
use crate::system::system::System;
use crate::util::{bit_reverse, BATCH_SIZE, SALT_SIZE, SIZE_F};

use crate::plonk::prover::fri_proof;

pub struct PolynomialBatch {
    pub name: String,
    pub addr_polynomials: usize,
    pub addr_leaves: usize,
    pub addr_transposed_leaves: usize,
    pub addr_digests: usize,
    pub addr_cap: usize,
    pub addr_salt: usize,
    pub leaf_length: usize,
    pub degree_log: usize,
    pub rate_bits: usize,
    pub blinding: bool,
    pub padding_length: usize,
}
impl PolynomialBatch {
    pub fn new(
        name: &str,
        sys: &mut System,
        addr_values: usize,
        degree: usize,
        num_kernels: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        transposed_input: bool,
        merged_input: Vec<MemCpy>,
    ) -> Self {
        info!("commitment: {:?}", name);
        let salt_size = if blinding { SALT_SIZE } else { 0 };

        let salt_id = format!("{}{}", "salt_poly_cpu", name);
        let digests_id = format!("{}{}", "digests", name);
        let cap_id = format!("{}{}", "cap", name);
        let leaves_id = format!("{}{}", "leaves", name);
        let transposed_leaves_id = format!("{}{}", "transposed_leaves", name);
        let polynomials_id = format!("{}{}", "polynomials", name);

        let padding_length =
            Tree::get_padding_length(degree << rate_bits, num_kernels + salt_size, cap_height);

        let addr_salt = if blinding {
            sys.mem
                .alloc(
                    &salt_id,
                    ((degree << rate_bits) + padding_length) * SALT_SIZE * SIZE_F,
                )
                .unwrap()
        } else {
            0
        };
        let addr_digest_buf = sys
            .mem
            .alloc(
                &digests_id,
                Tree::num_digests(degree << rate_bits, cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_cap_buf = sys
            .mem
            .alloc(
                &cap_id,
                Tree::num_caps(cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_leaves = sys
            .mem
            .alloc(
                &leaves_id,
                ((degree << rate_bits) + padding_length) * (num_kernels + salt_size) * SIZE_F,
            )
            .unwrap();
        let addr_transposed_leaves = sys
            .mem
            .alloc(
                &transposed_leaves_id,
                (degree << rate_bits) * (num_kernels + salt_size) * SIZE_F,
            )
            .unwrap();
        let addr_polynomials = sys
            .mem
            .alloc(&polynomials_id, degree * num_kernels * SIZE_F)
            .unwrap();
        let pb = Self {
            name: name.to_string(),
            addr_leaves: addr_leaves,
            addr_transposed_leaves: addr_transposed_leaves,
            addr_digests: addr_digest_buf,
            addr_cap: addr_cap_buf,
            addr_polynomials: addr_polynomials,
            addr_salt: addr_salt,
            leaf_length: num_kernels + salt_size,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
            padding_length,
        };

        pb.from_values(
            sys,
            addr_values,
            degree,
            num_kernels,
            rate_bits,
            blinding,
            cap_height,
            transposed_input,
            merged_input,
        );
        if blinding {
            sys.mem.free(&salt_id);
        }
        pb
    }

    // only for commitment addr, indicates CPU to ASIC copy
    pub fn new_alloc(
        name: &str,
        sys: &mut System,
        degree: usize,
        num_kernels: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
    ) -> Self {
        let salt_size: usize = if blinding { SALT_SIZE } else { 0 };

        let digests_id = format!("{}{}", "digests", name);
        let cap_id = format!("{}{}", "cap", name);
        let leaves_id = format!("{}{}", "leaves", name);
        let polynomials_id = format!("{}{}", "polynomials", name);

        let addr_digest_buf = sys
            .mem
            .alloc(
                &digests_id,
                Tree::num_digests(degree << rate_bits, cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_cap_buf = sys
            .mem
            .alloc(
                &cap_id,
                Tree::num_caps(cap_height) * Tree::DIGEST_LENGTH * SIZE_F,
            )
            .unwrap();
        let addr_leaves = sys
            .mem
            .alloc(
                &leaves_id,
                (degree << rate_bits) * (num_kernels + salt_size) * SIZE_F,
            )
            .unwrap();
        let addr_polynomials = sys
            .mem
            .alloc(&polynomials_id, degree * num_kernels * SIZE_F)
            .unwrap();

        PolynomialBatch {
            name: name.to_string(),
            addr_polynomials: addr_polynomials,
            addr_leaves: addr_leaves,
            addr_transposed_leaves: addr_leaves,
            addr_digests: addr_digest_buf,
            addr_cap: addr_cap_buf,
            addr_salt: 0,
            leaf_length: num_kernels + salt_size,
            degree_log: log2_strict(degree),
            rate_bits,
            blinding,
            padding_length: 0,
        }
    }

    pub fn _free(&self, sys: &mut System) {
        let digests_id = String::from("digests") + self.name.as_str();
        let cap_id = String::from("cap") + self.name.as_str();
        let leaves_id = String::from("leaves") + self.name.as_str();
        let polynomials_id = String::from("polynomials") + self.name.as_str();
        sys.mem.free(&leaves_id);
        sys.mem.free(&cap_id);
        sys.mem.free(&digests_id);
        sys.mem.free(&polynomials_id);
    }

    /// return (start, size(number of elements))
    pub fn get_lde_values_addr(&self, index: usize, step: usize) -> (usize, usize) {
        let index = index * step;
        let index = bit_reverse(index, self.degree_log + self.rate_bits);
        let addr_leaf = self.addr_transposed_leaves + index * self.leaf_length * SIZE_F;
        (
            addr_leaf,
            self.leaf_length - if self.blinding { SALT_SIZE } else { 0 },
        )
    }

    pub fn get_lde_values_packed(&self, index_start: usize, step: usize) -> Vec<(usize, usize)> {
        (0..BATCH_SIZE)
            .map(|i| self.get_lde_values_addr(index_start + i, step))
            .collect::<Vec<_>>()
    }

    pub fn from_values(
        &self,
        sys: &mut System,
        addr_values: usize,
        degree: usize,
        num_kernels: usize,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        transposed_input: bool,
        merged_input: Vec<MemCpy>,
    ) {
        let addr_coeffs = self.addr_polynomials;

        let mut ifft_kernel = Fft::new(FftConfig {
            lg_n: log2_strict(degree),
            k: num_kernels,
            direction: FftDirection::NN,
            addr_input: addr_values,
            addr_tmp: 1 << 60,
            addr_output: addr_coeffs,
            inverse: true,
            rate_bits: 0,
            coset: false,
            extension: 1,
            transposed_input: transposed_input,
        });
        for memcpy in &merged_input {
            ifft_kernel.prefetch.addr_trans(memcpy);
        }
        sys.run_once(&ifft_kernel);
        self.from_coeffs(sys, rate_bits, blinding, cap_height, degree, num_kernels);
    }

    fn from_coeffs(
        &self,
        sys: &mut System,
        rate_bits: usize,
        blinding: bool,
        cap_height: usize,
        mut degree: usize,
        num_kernels: usize,
    ) {
        let salt_size = if blinding { SALT_SIZE } else { 0 };
        let addr_coeffs = self.addr_polynomials;
        let addr_leaves = self.addr_leaves;
        let addr_digest_buf = self.addr_digests;
        let addr_cap_buf = self.addr_cap;
        let addr_salt = self.addr_salt;

        degree = degree << rate_bits;

        let mut padding_kernel = (0..num_kernels)
            .map(|i| {
                MemCpy::new(
                    MemCpyConfig {
                        addr_input: addr_leaves + i * (degree + self.padding_length) * SIZE_F,
                        addr_output: addr_leaves + i * degree * SIZE_F,
                        input_length: degree,
                    },
                    false,
                )
            })
            .collect::<Vec<_>>();

        let addr_addr_coeffs_tmp = sys
            .mem
            .alloc("coeffs_tmp", degree * num_kernels * SIZE_F)
            .unwrap();
        let mut coset_fft_kernel = Fft::new(FftConfig {
            lg_n: log2_strict(degree),
            k: num_kernels,
            direction: FftDirection::NN,
            addr_input: addr_coeffs,
            addr_output: addr_leaves,
            addr_tmp: addr_addr_coeffs_tmp,
            inverse: false,
            rate_bits,
            coset: true,
            extension: 1,
            transposed_input: false,
        });
        sys.mem.free("coeffs_tmp");

        if blinding {
            for i in 0..SALT_SIZE {
                padding_kernel.push(MemCpy::new(
                    MemCpyConfig {
                        addr_input: addr_salt + i * (degree + self.padding_length) * SIZE_F,
                        addr_output: addr_leaves + (num_kernels + i) * degree * SIZE_F,
                        input_length: degree,
                    },
                    false,
                ));
            }
        }

        coset_fft_kernel.prefetch.addr_trans_vec(&padding_kernel);
        coset_fft_kernel.drain.addr_trans_vec(&padding_kernel);

        sys.run_once(&coset_fft_kernel);

        let mut tree_kernel = Tree::new(TreeConfig {
            leaf_length: num_kernels + salt_size,
            cap_height,
            num_leaves: degree,
            addr_leaves: addr_leaves,
            addr_transposed_leaves: self.addr_transposed_leaves,
            addr_digest_buf: addr_digest_buf,
            addr_cap_buf,
            transposed_leaves: true,
        });
        padding_kernel
            .iter()
            .for_each(|memcpy| tree_kernel.prefetch.addr_trans(memcpy));
        sys.run_once(&tree_kernel);
    }

    pub fn prove_openings<F: RichField + Extendable<D>, const D: usize>(
        sys: &mut System,
        instance: &FriInstanceInfo<F, D>,
        oracles: &Vec<&Self>,
        challenger: &mut Challenger,
        fri_params: &FriParams,
        addr_zeta: usize,
        addr_zeta_g: usize,
    ) {
        info!("prove_openings");
        let addr_alpha = sys.mem.alloc("alpha", D * SIZE_F).unwrap();
        challenger.get_extension_challenge::<D>(sys, addr_alpha);
        let mut alpha = ReducingFactor::new(addr_alpha);

        let mut vec_ops = Vec::new();
        let mut addr_point = addr_zeta;

        let mut final_poly_degree = instance
            .batches
            .iter()
            .map(
                |FriBatchInfo {
                     point: _,
                     polynomials,
                 }| {
                    polynomials
                        .iter()
                        .map(|fri_poly| (1 << &oracles[fri_poly.oracle_index].degree_log))
                        .max()
                        .unwrap()
                },
            )
            .max()
            .unwrap();
        let addr_final_poly = sys
            .mem
            .alloc(
                "final_poly",
                (final_poly_degree << fri_params.config.rate_bits) * D * SIZE_F,
            )
            .unwrap();
        for FriBatchInfo {
            point: _,
            polynomials,
        } in &instance.batches
        {
            let degrees = polynomials
                .iter()
                .map(|fri_poly| (1 << &oracles[fri_poly.oracle_index].degree_log))
                .collect::<Vec<_>>();
            let addr_polys_coeff = polynomials
                .iter()
                .zip(degrees.iter())
                .map(|(fri_poly, degree)| {
                    &oracles[fri_poly.oracle_index].addr_polynomials
                        + fri_poly.polynomial_index * degree
                })
                .collect();
            let max_degree = degrees.iter().max().unwrap();
            let addr_composition_poly = sys
                .mem
                .alloc("composition_poly", max_degree * D * SIZE_F)
                .unwrap();
            vec_ops.extend(alpha.reduce_polys_base::<D>(
                sys,
                &addr_polys_coeff,
                &degrees,
                addr_composition_poly,
            ));
            let addr_quotient = sys.mem.alloc("quotient", max_degree * D * SIZE_F).unwrap();
            vec_ops.extend(divide_by_linear::<D>(
                sys,
                addr_point,
                addr_composition_poly,
                addr_quotient,
                *max_degree,
            ));
            vec_ops.extend(alpha.shift_poly::<D>(sys, addr_final_poly, final_poly_degree));
            vec_ops.extend(VecOpExtension::<D>::add(
                *max_degree,
                addr_quotient,
                addr_final_poly,
                addr_final_poly,
                VecOpSrc::VV,
                true,
            ));
            sys.mem.free("composition_poly");
            sys.mem.free("quotient");

            addr_point = addr_zeta_g;
        }

        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));

        let addr_final_coeff = sys
            .mem
            .alloc(
                "final_coeff",
                (final_poly_degree << fri_params.config.rate_bits) * D * SIZE_F,
            )
            .unwrap();
        // sys.run_once(MemCpy::new(MemCpyConfig {
        //     addr_input: addr_final_poly,
        //     addr_output: addr_final_coeff,
        //     input_length: final_poly_degree * D,
        // }));
        let addr_final_values = sys
            .mem
            .alloc("final_values", final_poly_degree * D * SIZE_F)
            .unwrap();

        final_poly_degree = final_poly_degree << fri_params.config.rate_bits;
        sys.run_once(&Fft::new(FftConfig {
            lg_n: log2_strict(final_poly_degree),
            k: 1,
            direction: FftDirection::NR,
            addr_input: addr_final_poly,
            addr_tmp: 1 << 60,
            addr_output: addr_final_values,
            inverse: false,
            rate_bits: fri_params.config.rate_bits,
            coset: true,
            extension: D,
            transposed_input: false,
        }));

        fri_proof::<D>(
            sys,
            oracles,
            addr_final_coeff,
            addr_final_values,
            final_poly_degree,
            challenger,
            fri_params,
        );
        sys.mem.free("final_poly");
    }

    pub fn num_layers(&self, cap_height: usize) -> usize {
        self.degree_log + self.rate_bits - cap_height
    }

    pub fn prove(&self, index: usize, cap_height: usize, addr_proof: usize) -> Vec<MemCpy> {
        let mut res = Vec::new();
        let num_layers = self.num_layers(cap_height);

        let digest_tree = {
            let tree_index = index >> num_layers;

            let tree_len = Tree::num_digests(1 << (self.degree_log + self.rate_bits), cap_height)
                >> cap_height;
            self.addr_digests + tree_len * tree_index * Tree::DIGEST_LENGTH * SIZE_F
        };

        let mut pair_index = index & ((1 << num_layers) - 1);
        let mut num_nodes = 0;
        for i in 0..num_layers {
            let parity = pair_index & 1;
            pair_index >>= 1;

            let siblings_index = pair_index << 1;
            let sibling_index = num_nodes + siblings_index + (1 - parity);
            num_nodes += 1 << (num_layers - i);
            res.push(MemCpy::new(
                MemCpyConfig {
                    addr_input: digest_tree + sibling_index * Tree::DIGEST_LENGTH * SIZE_F,
                    addr_output: addr_proof + i * Tree::DIGEST_LENGTH * SIZE_F,
                    input_length: Tree::DIGEST_LENGTH,
                },
                unsafe { ENABLE_CONFIG.tree },
            ));
        }
        res
    }
}

fn divide_by_linear<const D: usize>(
    sys: &mut System,
    addr_z: usize,
    addr_coeffs: usize,
    addr_res: usize,
    degree: usize,
) -> Vec<VecOpConfig> {
    let mut vec_ops = Vec::new();
    for i in 1..degree {
        let addr_acc_input = if i == 0 {
            0
        } else {
            addr_res + (i - 1) * SIZE_F
        };

        vec_ops.extend(VecOpExtension::<D>::mul(
            &mut sys.mem,
            1,
            addr_z,
            addr_acc_input,
            addr_res + (degree - i - 1) * SIZE_F,
            VecOpSrc::VS,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::add(
            1,
            addr_res + (degree - i - 1) * SIZE_F,
            addr_coeffs + (degree - i) * SIZE_F,
            addr_res + (degree - i - 1) * SIZE_F,
            VecOpSrc::VS,
            true,
        ));
    }
    vec_ops
}

pub struct ReducingFactor {
    addr_base: usize,
    count: u64,
}
impl ReducingFactor {
    pub const fn new(addr_base: usize) -> Self {
        Self {
            addr_base,
            count: 0,
        }
    }

    pub fn reduce_polys_base<const D: usize>(
        &mut self,
        sys: &mut System,
        addr_polys: &Vec<usize>,
        degree: &Vec<usize>,
        addr_res: usize,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let max_degree = degree.iter().max().unwrap();
        let addr_reduce_polys_base_tmp = sys
            .mem
            .alloc("reduce_polys_base_tmp", max_degree * D * SIZE_F)
            .unwrap();
        let addr_base_power = sys.mem.alloc("base_power", SIZE_F).unwrap();
        for (_, degree) in addr_polys.iter().zip(degree.iter()) {
            vec_ops.extend(VecOpExtension::<D>::scalar_mul(
                *degree,
                addr_polys[self.count as usize],
                addr_base_power,
                addr_reduce_polys_base_tmp,
                VecOpSrc::VS,
                false,
            ));
            vec_ops.extend(VecOpExtension::<D>::add(
                *degree,
                addr_reduce_polys_base_tmp,
                addr_res,
                addr_res,
                VecOpSrc::VV,
                true,
            ));
            vec_ops.push(VecOpConfig {
                vector_length: 1,
                addr_input_0: addr_base_power,
                addr_input_1: addr_base_power,
                addr_output: addr_base_power,
                op_type: VecOpType::MUL,
                op_src: VecOpSrc::VS,
                is_final_output: true,
            });
            self.count += 1;
        }
        sys.mem.free("reduce_polys_base_tmp");
        sys.mem.free("base_power");
        vec_ops
    }

    pub fn shift_poly<const D: usize>(
        &mut self,
        sys: &mut System,
        addr_p: usize,
        degree: usize,
    ) -> Vec<VecOpConfig> {
        let mut vec_ops = Vec::new();
        let addr_base_exp = sys.mem.alloc("base_exp", D * SIZE_F).unwrap();
        vec_ops.extend(VecOpExtension::<D>::exp_u64(
            &mut sys.mem,
            self.addr_base,
            addr_base_exp,
            false,
        ));
        vec_ops.extend(VecOpExtension::<D>::mul(
            &mut sys.mem,
            degree,
            addr_p,
            addr_base_exp,
            addr_p,
            VecOpSrc::VS,
            true,
        ));
        sys.mem.free("base_exp");
        self.count = 0;
        vec_ops
    }
}
