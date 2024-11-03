use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::memory_copy::{MemCpy, MemCpyConfig};
use crate::kernel::vector_chain::VectorChain;
use crate::kernel::vector_operation::{VecOpExtension, VecOpSrc};
use crate::plonk::oracle::PolynomialBatch;
use crate::system::system::System;
use crate::util::SIZE_F;

pub struct StarkOpeningSet<const D: usize> {
    pub addr_local_values: usize,
    pub local_values_length: usize,
    pub addr_next_values: usize,
    pub next_values_length: usize,
    pub addr_quotient_polys: usize,
    pub quotient_polys_length: usize,
}

impl<const D: usize> StarkOpeningSet<D> {
    pub fn new(
        sys: &mut System,
        addr_zeta: usize,
        addr_g: usize,
        trace_commitment: &PolynomialBatch,
        quotient_commitment: &PolynomialBatch,
    ) -> Self {
        let mut vec_ops = Vec::new();
        // zeta * g
        let addr_zeta_next = sys.mem.alloc("zeta_next", D * SIZE_F).unwrap();
        vec_ops.extend(VecOpExtension::<D>::scalar_mul(
            1,
            addr_zeta,
            addr_g,
            addr_zeta_next,
            VecOpSrc::VS,
            false,
        ));
        sys.mem.preload(addr_zeta, D);
        sys.mem.preload(addr_zeta_next, D);

        sys.run_once(&VectorChain::new(vec_ops, &sys.mem));

        let eval_commitment =
            |sys: &mut System, z: Vec<usize>, c: &PolynomialBatch, res: Vec<usize>| {
                let addr_poly = c.addr_polynomials;
                let degree = 1 << c.degree_log;
                let num_polys = c.leaf_length;

                for addr_res in res.iter() {
                    sys.mem.preload(*addr_res, num_polys * D);
                }

                let batch_size = unsafe { ARCH_CONFIG.num_elems() / num_polys };
                for i_start in (0..degree).step_by(batch_size).rev() {
                    let mut vec_ops = Vec::new();
                    let mut mks = Vec::new();
                    let i_len = batch_size.min(degree - i_start);
                    for j in 0..num_polys {
                        let mut m = MemCpy::new(
                            MemCpyConfig {
                                addr_input: addr_poly + j * degree * SIZE_F + i_start * SIZE_F,
                                addr_output: 0,
                                input_length: i_len,
                            },
                            unsafe { ENABLE_CONFIG.other },
                        );
                        m.drain.clear();
                        m.write_request.clear();
                        mks.push(m);
                    }
                    sys.run_vec(mks);

                    for i in (i_start..i_start + i_len).rev() {
                        for (addr_z, addr_res) in z.iter().zip(res.iter()) {
                            // acc * z
                            vec_ops.extend(VecOpExtension::<D>::mul(
                                &mut sys.mem,
                                num_polys,
                                *addr_res,
                                *addr_z,
                                *addr_res,
                                VecOpSrc::VS,
                                false,
                            ));
                            // _ + c
                            vec_ops.extend(VecOpExtension::<D>::add(
                                num_polys,
                                *addr_res,
                                0,
                                *addr_res,
                                VecOpSrc::VV,
                                i == i_start,
                            ));
                        }
                    }
                    sys.run_once(&VectorChain::new(vec_ops, &sys.mem));
                }

                for addr_res in res.iter() {
                    sys.mem.unpreload(*addr_res);
                }
            };

        let addr_local_values = sys
            .mem
            .alloc("local_values", trace_commitment.leaf_length * D * SIZE_F)
            .unwrap();
        let addr_next_values = sys
            .mem
            .alloc("next_values", trace_commitment.leaf_length * D * SIZE_F)
            .unwrap();
        let addr_quotient_polys = sys
            .mem
            .alloc(
                "quotient_polys",
                quotient_commitment.leaf_length * D * SIZE_F,
            )
            .unwrap();

        eval_commitment(
            sys,
            vec![addr_zeta, addr_zeta_next],
            trace_commitment,
            vec![addr_local_values, addr_next_values],
        );
        eval_commitment(
            sys,
            vec![addr_zeta],
            quotient_commitment,
            vec![addr_quotient_polys],
        );

        sys.mem.clear_preload();
        sys.mem.free("zeta_next");

        StarkOpeningSet {
            addr_local_values,
            local_values_length: trace_commitment.leaf_length,
            addr_next_values,
            next_values_length: trace_commitment.leaf_length,
            addr_quotient_polys,
            quotient_polys_length: quotient_commitment.leaf_length,
        }
    }

    pub fn destroy(&self, sys: &mut System) {
        sys.mem.free("local_values");
        sys.mem.free("next_values");
        sys.mem.free("quotient_polys");
    }
}
