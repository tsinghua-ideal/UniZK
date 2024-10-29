use plonky2::field::extension::Extendable;
use plonky2::hash::hash_types::RichField;
use plonky2::plonk::circuit_data::CommonCircuitData;

use crate::config::arch_config::ARCH_CONFIG;
use crate::kernel::transpose::{Transpose, TransposeConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpExtension, VecOpSrc};
use crate::memory::memory_allocator::MemAlloc;
use crate::plonk::oracle::PolynomialBatch;
use crate::system::system::System;
use crate::util::SIZE_F;

pub struct OpeningSet<const D: usize> {
    pub addr_constants: usize,
    pub constants_length: usize,
    pub addr_plonk_sigmas: usize,
    pub plonk_sigmas_length: usize,
    pub addr_wires: usize,
    pub wires_length: usize,
    pub addr_plonk_zs: usize,
    pub plonk_zs_length: usize,
    pub addr_plonk_zs_next: usize,
    pub plonk_zs_next_length: usize,
    pub addr_partial_products: usize,
    pub partial_products_length: usize,
    pub addr_quotient_polys: usize,
    pub quotient_polys_length: usize,
    pub addr_lookup_zs: usize,
    pub lookup_zs_length: usize,
    pub addr_lookup_zs_next: usize,
    pub lookup_zs_next_length: usize,
}

impl<const D: usize> OpeningSet<D> {
    const SIZE_FE: usize = D * SIZE_F;
    pub fn new<F: RichField + Extendable<D>>(
        sys: &mut System,
        addr_zeta: usize,
        addr_zeta_g: usize,
        constants_sigmas_commitment: &PolynomialBatch,
        wires_commitment: &PolynomialBatch,
        zs_partial_products_lookup_commitment: &PolynomialBatch,
        quotient_polys_commitment: &PolynomialBatch,
        common_data: &CommonCircuitData<F, D>,
    ) -> Self {
        let addr_constants_sigmas_eval = sys
            .mem
            .alloc(
                "constants_sigmas_eval",
                constants_sigmas_commitment.leaf_length * Self::SIZE_FE,
            )
            .unwrap();
        let addr_zs_partial_products_lookup_eval = sys
            .mem
            .alloc(
                "zs_partial_products_lookup_eval",
                zs_partial_products_lookup_commitment.leaf_length * Self::SIZE_FE,
            )
            .unwrap();
        let addr_zs_partial_products_lookup_next_eval = sys
            .mem
            .alloc(
                "zs_partial_products_lookup_next_eval",
                zs_partial_products_lookup_commitment.leaf_length * Self::SIZE_FE,
            )
            .unwrap();
        let addr_quotient_polys = sys
            .mem
            .alloc(
                "quotient_polys",
                quotient_polys_commitment.leaf_length * Self::SIZE_FE,
            )
            .unwrap();
        let addr_wires_eval = sys
            .mem
            .alloc("wires_eval", wires_commitment.leaf_length * Self::SIZE_FE)
            .unwrap();

        Self::eval_commitment(
            sys,
            addr_zeta,
            constants_sigmas_commitment,
            addr_constants_sigmas_eval,
        );
        Self::eval_commitment(
            sys,
            addr_zeta,
            zs_partial_products_lookup_commitment,
            addr_zs_partial_products_lookup_eval,
        );
        Self::eval_commitment(
            sys,
            addr_zeta_g,
            zs_partial_products_lookup_commitment,
            addr_zs_partial_products_lookup_next_eval,
        );
        Self::eval_commitment(
            sys,
            addr_zeta,
            quotient_polys_commitment,
            addr_quotient_polys,
        );
        Self::eval_commitment(sys, addr_zeta, wires_commitment, addr_wires_eval);

        Self {
            addr_constants: addr_constants_sigmas_eval,
            constants_length: common_data.num_constants,
            addr_plonk_sigmas: addr_constants_sigmas_eval
                + common_data.num_constants * Self::SIZE_FE,
            plonk_sigmas_length: common_data.config.num_routed_wires,
            addr_wires: addr_wires_eval,
            wires_length: wires_commitment.leaf_length,
            addr_plonk_zs: addr_zs_partial_products_lookup_eval,
            plonk_zs_length: common_data.config.num_challenges,
            addr_plonk_zs_next: addr_zs_partial_products_lookup_next_eval,
            plonk_zs_next_length: common_data.config.num_challenges,
            addr_partial_products: addr_zs_partial_products_lookup_eval
                + common_data.config.num_challenges * Self::SIZE_FE,
            partial_products_length: common_data.num_partial_products
                * common_data.config.num_challenges,
            addr_quotient_polys: addr_quotient_polys,
            quotient_polys_length: quotient_polys_commitment.leaf_length,
            addr_lookup_zs: addr_zs_partial_products_lookup_eval
                + common_data.config.num_challenges
                    * (1 + common_data.num_partial_products)
                    * Self::SIZE_FE,
            lookup_zs_length: zs_partial_products_lookup_commitment.leaf_length
                - common_data.config.num_challenges * (1 + common_data.num_partial_products),
            addr_lookup_zs_next: addr_zs_partial_products_lookup_next_eval
                + common_data.config.num_challenges
                    * (1 + common_data.num_partial_products)
                    * Self::SIZE_FE,
            lookup_zs_next_length: zs_partial_products_lookup_commitment.leaf_length
                - common_data.config.num_challenges * (1 + common_data.num_partial_products),
        }
    }
    pub fn _free(&self, mem: &mut MemAlloc) {
        mem.free("constants_sigmas_eval");
        mem.free("zs_partial_products_lookup_eval");
        mem.free("zs_partial_products_lookup_next_eval");
        mem.free("quotient_polys");
        mem.free("wires_eval");
    }

    fn eval_commitment(
        sys: &mut System,
        addr_zeta: usize,
        commitment: &PolynomialBatch,
        addr_res: usize,
    ) {
        let addr_poly = commitment.addr_polynomials;
        let degree = 1 << commitment.degree_log;
        let num_polys = commitment.leaf_length;

        let addr_eval_commitment_tmp = sys
            .mem
            .alloc("eval_commitment_tmp", unsafe {
                ARCH_CONFIG.array_length * num_polys * SIZE_F
            })
            .unwrap();
        let chunk_length = unsafe { ARCH_CONFIG.num_elems() / num_polys };
        for i in (0..degree).rev().step_by(chunk_length) {
            let mut vec_ops = Vec::new();
            let mut eval_commitment_trans_kernel = Transpose::new(TransposeConfig {
                addr_input: addr_poly,
                addr_output: addr_eval_commitment_tmp,
                width: degree,
                height: num_polys,
                reverse: false,
                extension: 1,
                start: i + 1 - chunk_length.min(i + 1),
                end: i + 1,
            });
            eval_commitment_trans_kernel.drain.clear();
            eval_commitment_trans_kernel.write_request.clear();
            sys.run_once(&eval_commitment_trans_kernel);
            for _j in i + 1 - chunk_length.min(i + 1)..=i {
                // acc * x
                vec_ops.extend(VecOpExtension::<D>::mul(
                    &mut sys.mem,
                    num_polys,
                    addr_res,
                    addr_zeta,
                    addr_res,
                    VecOpSrc::VS,
                    false,
                ));
                // _ + c
                vec_ops.extend(VecOpExtension::<D>::add(
                    num_polys,
                    addr_res,
                    0,
                    addr_res,
                    VecOpSrc::VV,
                    true,
                ));
            }
            sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
        }

        sys.mem.free("eval_commitment_tmp");
    }
}
