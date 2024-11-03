use log::debug;
use num::integer::Roots;
use plonky2::util::{log2_ceil, log2_strict};

use crate::config::arch_config::ARCH_CONFIG;
use crate::config::enable_config::ENABLE_CONFIG;
use crate::kernel::kernel::Kernel;
use crate::trace::trace::{Fetch, FetchType, Request};
use crate::util::{bit_reverse, ceil_div_usize, D, SIZE_F};
use plonky2::field::goldilocks_field::GoldilocksField as F;
use plonky2::field::types::Field;
use std::cmp::min;
use std::mem::swap;

use super::vector_operation::VecOpExtension;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FftDirection {
    NN, // natural order input and natural order output
    NR, // natural order input and reverse order output
}
#[derive(Clone, Copy, Debug)]
pub struct FftConfig {
    pub lg_n: usize, // log2 of the size of the FFT
    pub k: usize,    // number of FFTs
    pub direction: FftDirection,
    pub transposed_input: bool,
    pub addr_input: usize,  // address in memory of the input data
    pub addr_tmp: usize,    // if addr_input should be changed, use this address
    pub addr_output: usize, // address in memory of the output data
    pub inverse: bool,      // whether the FFT is inverse
    pub rate_bits: usize,   // number of bits for LDE, 0 for no LDE
    pub coset: bool,        // whether the FFT is coset
    pub extension: usize,   // extension elements
}

impl FftConfig {
    pub fn get_log_plane_length(&self) -> usize {
        let mut log2_ex = log2_ceil(self.extension);
        if log2_ex > 0 {
            log2_ex -= 1;
        }
        (unsafe { ARCH_CONFIG.array_length } >> 1) - 1 - log2_ex
    }

    pub fn get_plane_length(&self) -> usize {
        1 << self.get_log_plane_length()
    }

    pub fn get_num_planes(&self) -> usize {
        unsafe {
            ARCH_CONFIG.active_buf_size() * 1024
                / (8 * self.extension)
                / (self.get_plane_length() * self.get_plane_length())
        }
    }
}

// todo we need an extra round to perform the twiddle mul between dimensions
pub struct Fft {
    pub config: FftConfig,
    pub prefetch: Fetch,
    pub read_request: Request,
    pub write_request: Request,
    pub drain: Fetch,
}

impl Kernel for Fft {
    fn create_prefetch(&mut self) {
        let num_planes = self.config.get_num_planes();
        let plane_length = self.config.get_plane_length();
        let lg_plane_length = self.config.get_log_plane_length();
        let num_plane_elems = plane_length * plane_length;
        let ntt_length = 1 << self.config.lg_n;
        let num_pipes = unsafe { ARCH_CONFIG.array_length };

        if ntt_length <= plane_length {
            let mut idx = Vec::new();
            let kernel_step = num_pipes * num_planes;
            for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                let mut idx_chunk = Vec::new();
                for col in 0..ntt_length {
                    let req = (kernel_chunk..min(kernel_chunk + kernel_step, self.config.k))
                        .map(|row| row * ntt_length + col)
                        .collect();
                    idx_chunk.push(req);
                }
                idx.push(idx_chunk);
            }

            for idx_once in idx {
                let addr = self.idx_to_addri(idx_once, 0);
                self.prefetch.push(addr);
            }
        } else if ntt_length <= num_plane_elems {
            let residual_length = ntt_length / plane_length;
            let mut idx = Vec::new();

            for kernel_chunk in (0..self.config.k).step_by(num_planes) {
                let mut idx_chunk = Vec::new();
                for kernel_idx in kernel_chunk..min(kernel_chunk + num_planes, self.config.k) {
                    for col_chunk in (0..plane_length).step_by(num_pipes) {
                        for row in 0..residual_length {
                            let req = (col_chunk..min(col_chunk + num_pipes, plane_length))
                                .map(|col| kernel_idx * ntt_length + row * plane_length + col)
                                .collect();
                            idx_chunk.push(req);
                        }
                    }
                }
                idx.push(idx_chunk);
            }

            for idx_once in idx {
                let addr = self.idx_to_addri(idx_once, 0);
                self.prefetch.push(addr);
            }
        } else {
            let num_dims = ceil_div_usize(self.config.lg_n, lg_plane_length);
            let residual_length = 1 << (self.config.lg_n - (num_dims - 1) * lg_plane_length);
            let num_rounds = ceil_div_usize(num_dims, 2);

            let mut now_dim = (num_dims - 1) as u32;

            for round in 0..num_rounds {
                let offset = round % 2 != 0;
                let plane_height = if round == 0 {
                    residual_length
                } else {
                    plane_length
                };
                let para_k = if self.config.transposed_input && round == 0 {
                    self.config.k.min(self.config.get_num_planes().sqrt())
                } else {
                    1
                };
                let num_planes = ntt_length / plane_length / plane_height;
                let plane_step = (self.config.get_num_planes() / para_k).min(num_planes);
                let kernel_step = self.config.get_num_planes() / plane_step;

                for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                    let mut idx = Vec::new();
                    for plane_chunk in (0..num_planes).step_by(plane_step) {
                        let mut idx_chunk = Vec::new();
                        for kernel_idx in
                            kernel_chunk..min(kernel_chunk + kernel_step, self.config.k)
                        {
                            for row in 0..plane_height {
                                for col_chunk in (0..plane_length).step_by(num_pipes) {
                                    for plane_idx in
                                        plane_chunk..min(plane_chunk + plane_step, num_planes)
                                    {
                                        let req: Vec<usize> = (col_chunk
                                            ..min(col_chunk + num_pipes, plane_length))
                                            .map(|col| {
                                                let addr_base: usize = if round == 0 {
                                                    plane_idx
                                                } else {
                                                    plane_idx / plane_length.pow(now_dim - 1)
                                                        * plane_length.pow(now_dim + 1)
                                                        + plane_idx % plane_length.pow(now_dim - 1)
                                                };
                                                kernel_idx * ntt_length
                                                    + addr_base
                                                    + row * plane_length.pow(now_dim)
                                                    + col * plane_length.pow(now_dim - 1)
                                            })
                                            .collect();
                                        idx_chunk.push(req);
                                    }
                                }
                            }
                        }
                        // if round > 0 {
                        //     println!(
                        //         "idx_chunk elems: {:?}",
                        //         idx_chunk.iter().map(|x| x.len()).sum::<usize>()
                        //     );
                        // }
                        idx.push(idx_chunk);
                    }

                    for idx_once in idx {
                        let addr = self.idx_to_addr(idx_once, offset, round);
                        self.prefetch.push(addr);
                    }
                }
                if round < num_rounds - 1 {
                    now_dim -= if now_dim % 2 == 1 { 2 } else { 1 };
                }
            }
        }
        self.prefetch.delay = vec![0; self.prefetch.len()];
        self.prefetch.interval = 0.;
        self.prefetch.systolic = true;
    }

    fn create_read_request(&mut self) {
        let num_planes = self.config.get_num_planes();
        let plane_length = self.config.get_plane_length();
        let lg_plane_length = self.config.get_log_plane_length();
        let num_plane_elems = plane_length * plane_length;
        let ntt_length = 1 << self.config.lg_n;
        let num_pipes = unsafe { ARCH_CONFIG.array_length };

        if ntt_length <= plane_length {
            let kernel_step = num_pipes * num_planes;
            for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                let mut request = Vec::new();
                for col in 0..ntt_length {
                    let req = (kernel_chunk..min(kernel_chunk + kernel_step, self.config.k))
                        .map(|row| row * ntt_length + col)
                        .map(|x| x * SIZE_F * self.config.extension + self.config.addr_input)
                        .collect();
                    request.push(req);
                }
                self.read_request.push(request);
            }
        } else if ntt_length <= num_plane_elems {
            let residual_length = ntt_length / plane_length;
            for kernel_chunk in (0..self.config.k).step_by(num_planes) {
                let mut request = Vec::new();
                for kernel_idx in kernel_chunk..min(kernel_chunk + num_planes, self.config.k) {
                    for col_chunk in (0..plane_length).step_by(num_pipes) {
                        for row in 0..residual_length {
                            let req = (col_chunk..min(col_chunk + num_pipes, plane_length))
                                .map(|col| kernel_idx * ntt_length + row * plane_length + col)
                                .map(|x| {
                                    x * SIZE_F * self.config.extension + self.config.addr_input
                                })
                                .collect();
                            request.push(req);
                        }
                    }
                }
                self.read_request.push(request);
            }
        } else {
            let num_dims = ceil_div_usize(self.config.lg_n, lg_plane_length);
            let residual_length = 1 << (self.config.lg_n - (num_dims - 1) * lg_plane_length);
            let num_rounds = ceil_div_usize(num_dims, 2);

            let mut now_dim = (num_dims - 1) as u32;

            for round in 0..num_rounds {
                let offset = if round % 2 == 0 {
                    self.config.addr_input
                } else {
                    self.config.addr_output
                };
                let plane_height = if round == 0 {
                    residual_length
                } else {
                    plane_length
                };
                let para_k = if self.config.transposed_input && round == 0 {
                    self.config.k.min(self.config.get_num_planes().sqrt())
                } else {
                    1
                };
                let num_planes = ntt_length / plane_length / plane_height;
                let plane_step = (self.config.get_num_planes() / para_k).min(num_planes);
                let kernel_step = self.config.get_num_planes() / plane_step;
                for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                    for plane_chunk in (0..num_planes).step_by(plane_step) {
                        let mut request = Vec::new();
                        for kernel_idx in
                            kernel_chunk..min(kernel_chunk + kernel_step, self.config.k)
                        {
                            let plane_height = if round == 0 {
                                residual_length
                            } else {
                                plane_length
                            };
                            let num_planes = ntt_length / plane_length / plane_height;
                            for plane_idx in plane_chunk..min(plane_chunk + plane_step, num_planes)
                            {
                                for col_chunk in (0..plane_length).step_by(num_pipes) {
                                    for row in 0..plane_height {
                                        let req = (col_chunk
                                            ..min(col_chunk + num_pipes, plane_length))
                                            .map(|col| {
                                                let addr_base = if round == 0 {
                                                    plane_idx
                                                } else {
                                                    plane_idx / plane_length.pow(now_dim - 1)
                                                        * plane_length.pow(now_dim + 1)
                                                        + plane_idx % plane_length.pow(now_dim - 1)
                                                };
                                                addr_base
                                                    + kernel_idx * ntt_length
                                                    + row * plane_length.pow(now_dim)
                                                    + col * plane_length.pow(now_dim - 1)
                                            })
                                            .map(|x| x * SIZE_F * self.config.extension + offset)
                                            .collect();
                                        request.push(req);
                                    }
                                }
                            }
                        }
                        self.read_request.push(request);
                    }
                }

                if round < num_rounds - 1 {
                    now_dim -= if now_dim % 2 == 1 { 2 } else { 1 };
                }
            }
        }
    }

    fn create_write_request(&mut self) {
        let num_planes = self.config.get_num_planes();
        let plane_length = self.config.get_plane_length();
        let lg_plane_length = self.config.get_log_plane_length();
        let num_plane_elems = plane_length * plane_length;
        let ntt_length = 1 << self.config.lg_n;
        let num_pipes = unsafe { ARCH_CONFIG.array_length };

        if ntt_length <= plane_length {
            let kernel_step = num_pipes * num_planes;
            for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                let mut request = Vec::new();
                for col in 0..ntt_length {
                    let req = (kernel_chunk..min(kernel_chunk + kernel_step, self.config.k))
                        .map(|row| row * ntt_length + bit_reverse(col, self.config.lg_n))
                        .map(|x| x * SIZE_F * self.config.extension + self.config.addr_output)
                        .collect();
                    request.push(req);
                }
                self.write_request.push(request);
            }
        } else if ntt_length <= num_plane_elems {
            let residual_length = ntt_length / plane_length;
            for kernel_chunk in (0..self.config.k).step_by(num_planes) {
                let mut request = Vec::new();
                for kernel_idx in kernel_chunk..min(kernel_chunk + num_planes, self.config.k) {
                    for row_chunk in (0..residual_length).step_by(num_pipes) {
                        for col in 0..plane_length {
                            let req = (row_chunk..min(row_chunk + num_pipes, residual_length))
                                .map(|row| {
                                    kernel_idx * ntt_length
                                        + bit_reverse(col, lg_plane_length) * residual_length
                                        + row
                                })
                                .map(|x| {
                                    x * SIZE_F * self.config.extension + self.config.addr_output
                                })
                                .collect();
                            request.push(req);
                        }
                    }
                }
                self.write_request.push(request);
            }
        } else {
            let num_dims = ceil_div_usize(self.config.lg_n, lg_plane_length);
            let residual_length = 1 << (self.config.lg_n - (num_dims - 1) * lg_plane_length);
            let num_rounds = ceil_div_usize(num_dims, 2);

            let mut now_dim = (num_dims - 1) as u32;

            for round in 0..num_rounds {
                let addr_offset = if round % 2 == 1 {
                    self.config.addr_input
                } else {
                    self.config.addr_output
                };
                let lg_plane_height = if round == 0 {
                    self.config.lg_n - (num_dims - 1) * lg_plane_length
                } else {
                    lg_plane_length
                };
                let plane_height: usize = if round == 0 {
                    residual_length
                } else {
                    plane_length
                };
                let para_k = if self.config.transposed_input && round == 0 {
                    self.config.k.min(self.config.get_num_planes().sqrt())
                } else {
                    1
                };

                let num_planes = ntt_length / plane_length / plane_height;

                let factor = if now_dim as usize == num_dims - 1 {
                    1
                } else {
                    plane_length / residual_length
                };
                let plane_offset = self.plane_offset((num_dims - 2) as u32, factor);
                let plane_step = (self.config.get_num_planes() / para_k).min(plane_offset);
                let kernel_step = self.config.get_num_planes() / plane_step;

                for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                    for plane_chunk in (0..num_planes).step_by(plane_step) {
                        let mut request = Vec::new();
                        for kernel_idx in
                            kernel_chunk..min(kernel_chunk + kernel_step, self.config.k)
                        {
                            for plane_idx in plane_chunk..min(plane_chunk + plane_step, num_planes)
                            {
                                if now_dim as usize == num_dims - 1 && num_dims % 2 == 1 {
                                    for col_chunk in (0..plane_length).step_by(num_pipes) {
                                        for row in 0..plane_height {
                                            let req = (col_chunk
                                                ..min(col_chunk + num_pipes, plane_length))
                                                .map(|col| {
                                                    let addr_1d = bit_reverse(row, lg_plane_height)
                                                        * plane_length
                                                        + col;
                                                    let addr = plane_idx + addr_1d * plane_offset;
                                                    kernel_idx * ntt_length + addr
                                                })
                                                .map(|x| {
                                                    x * SIZE_F * self.config.extension + addr_offset
                                                })
                                                .collect();
                                            request.push(req);
                                        }
                                    }
                                } else {
                                    for row_chunk in (0..plane_height).step_by(num_pipes) {
                                        for col in 0..plane_length {
                                            let req = (row_chunk
                                                ..min(row_chunk + num_pipes, plane_height))
                                                .map(|row| {
                                                    let addr_1d = bit_reverse(col, lg_plane_length)
                                                        * plane_height
                                                        + row;
                                                    let addr = plane_idx + addr_1d * plane_offset;
                                                    kernel_idx * ntt_length + addr
                                                })
                                                .map(|x| {
                                                    x * SIZE_F * self.config.extension + addr_offset
                                                })
                                                .collect();
                                            request.push(req);
                                        }
                                    }
                                }
                            }
                        }
                        self.write_request.push(request);
                    }
                }
                if round < num_rounds - 1 {
                    now_dim -= if now_dim % 2 == 1 { 2 } else { 1 };
                }
            }
        }
    }

    fn create_drain(&mut self) {
        let is_nr = self.config.direction == FftDirection::NR;

        let num_planes = self.config.get_num_planes();
        let plane_length = self.config.get_plane_length();
        let lg_plane_length = self.config.get_log_plane_length();
        let num_plane_elems = plane_length * plane_length;
        let ntt_length = 1 << self.config.lg_n;
        let num_pipes = unsafe { ARCH_CONFIG.array_length };

        if ntt_length <= plane_length {
            let mut idx = Vec::new();

            let kernel_step = num_pipes * num_planes;

            for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                let mut idx_chunk = Vec::new();
                for col in 0..ntt_length {
                    let req = (kernel_chunk..min(kernel_chunk + kernel_step, self.config.k))
                        .map(|row| row * ntt_length + col)
                        .collect();
                    idx_chunk.push(req);
                }
                idx.push(idx_chunk);
            }

            for idx_once in idx {
                let idx_once_tmp = if is_nr {
                    self.idx_bit_reverse(idx_once)
                } else {
                    idx_once
                };
                let addr = self.idx_to_addro(idx_once_tmp);
                self.drain.push(addr);
            }
        } else if ntt_length <= num_plane_elems {
            let residual_length = ntt_length / plane_length;
            let mut idx = Vec::new();

            for kernel_chunk in (0..self.config.k).step_by(num_planes) {
                let mut idx_chunk = Vec::new();
                for kernel_idx in kernel_chunk..min(kernel_chunk + num_planes, self.config.k) {
                    for row_chunk in (0..residual_length).step_by(num_pipes) {
                        for col in 0..plane_length {
                            let req = (row_chunk..min(row_chunk + num_pipes, residual_length))
                                .map(|row| kernel_idx * ntt_length + col * residual_length + row)
                                .collect();
                            idx_chunk.push(req);
                        }
                    }
                }
                idx.push(idx_chunk);
            }

            for idx_once in idx {
                let idx_once_tmp = if is_nr {
                    self.idx_bit_reverse(idx_once)
                } else {
                    idx_once
                };
                let addr = self.idx_to_addro(idx_once_tmp);
                self.drain.push(addr);
            }
        } else {
            let num_dims = ceil_div_usize(self.config.lg_n, lg_plane_length);
            let residual_length = 1 << (self.config.lg_n - (num_dims - 1) * lg_plane_length);
            let num_rounds = ceil_div_usize(num_dims, 2);

            let mut now_dim = (num_dims - 1) as u32;
            for round in 0..num_rounds {
                let offset = round % 2 == 0;
                let plane_height: usize = if round == 0 {
                    residual_length
                } else {
                    plane_length
                };
                let para_k = if self.config.transposed_input && round == 0 {
                    self.config.k.min(self.config.get_num_planes().sqrt())
                } else {
                    1
                };

                let num_planes = ntt_length / plane_length / plane_height;

                let factor = if now_dim as usize == num_dims - 1 {
                    1
                } else {
                    plane_length / residual_length
                };
                let plane_offset = self.plane_offset((num_dims - 2) as u32, factor);
                let plane_step = (self.config.get_num_planes() / para_k).min(plane_offset);
                let kernel_step = self.config.get_num_planes() / plane_step;

                for kernel_chunk in (0..self.config.k).step_by(kernel_step) {
                    let mut idx = Vec::new();
                    for plane_chunk in (0..num_planes).step_by(plane_step) {
                        let mut idx_chunk = Vec::new();
                        for kernel_idx in
                            kernel_chunk..min(kernel_chunk + kernel_step, self.config.k)
                        {
                            for row_chunk in (0..plane_height).step_by(num_pipes) {
                                for col in 0..plane_length {
                                    for plane_idx in
                                        plane_chunk..min(plane_chunk + plane_step, num_planes)
                                    {
                                        let req: Vec<usize> = (row_chunk
                                            ..min(row_chunk + num_pipes, plane_height))
                                            .map(|row| {
                                                let addr_1d = if now_dim as usize == num_dims - 1 {
                                                    row * plane_length + col
                                                } else {
                                                    col * plane_height + row
                                                };
                                                let addr = plane_idx + addr_1d * plane_offset;
                                                kernel_idx * ntt_length + addr
                                            })
                                            .collect();
                                        idx_chunk.push(req);
                                    }
                                }
                            }
                        }
                        // if round > 0 {
                        //     println!(
                        //         "idx_chunk elems drain: {:?}",
                        //         idx_chunk.iter().map(|x| x.len()).sum::<usize>()
                        //     );
                        // }
                        idx.push(idx_chunk);
                    }

                    for idx_once in idx {
                        let idx_once: Vec<Vec<usize>> = if is_nr && round == num_rounds - 1 {
                            self.idx_bit_reverse(idx_once)
                        } else {
                            idx_once
                        };
                        let addr = self.idx_to_addr(idx_once, offset, round);
                        self.drain.push(addr);
                    }
                }
                if round < num_rounds - 1 {
                    now_dim -= if now_dim % 2 == 1 { 2 } else { 1 };
                }
            }
        }
        self.drain.systolic = true;
        let mul_delay = if self.config.extension == 1 {
            2
        } else {
            VecOpExtension::<D>::mul_delay()
        };
        self.drain.delay = vec![
            lg_plane_length * mul_delay * 2
                + plane_length * plane_length / num_pipes;
            self.drain.len()
        ];
        self.drain.interval = if self.config.extension == 1 {
            0.5
        } else {
            VecOpExtension::<D>::mul_delay() as f32 / 2.0
        };
    }

    fn get_prefetch(&self) -> Fetch {
        self.prefetch.clone()
    }
    fn get_drain(&self) -> Fetch {
        self.drain.clone()
    }
    fn get_read_request(&self) -> Request {
        self.read_request.clone()
    }
    fn get_write_request(&self) -> Request {
        self.write_request.clone()
    }
    fn log(&self) {
        debug!("Fft Kernel config: {:?}", self.config);
    }

    fn get_computation(&self) -> usize {
        self.config.k * (self.config.lg_n * (1 << self.config.lg_n) / 2)
    }
    fn get_kernel_type(&self) -> String {
        String::from("FFT")
    }
}

impl Fft {
    pub fn new(config: FftConfig) -> Fft {
        assert!(config.extension > 0);
        let mut fft = Fft {
            config,
            prefetch: Fetch::new(FetchType::Read),
            read_request: Request::new(),
            write_request: Request::new(),
            drain: Fetch::new(FetchType::Write),
        };
        let lg_plane_length = fft.config.get_log_plane_length();
        let num_dims = ceil_div_usize(fft.config.lg_n, lg_plane_length);
        let num_rounds = ceil_div_usize(num_dims, 2);
        assert_ne!(
            fft.config.addr_input, fft.config.addr_tmp,
            "addr_input and addr_tmp should be different"
        );
        if num_rounds % 2 == 0 {
            swap(&mut fft.config.addr_output, &mut fft.config.addr_tmp);
        }
        fft.log();
        if unsafe { ENABLE_CONFIG.fft } {
            fft.init()
        }
        fft
    }
    fn idx_to_addri(&self, idx: Vec<Vec<usize>>, round: usize) -> Vec<Vec<usize>> {
        let n = 1 << self.config.lg_n;
        let addr_input = if round == 0 {
            self.config.addr_input
        } else {
            self.config.addr_tmp
        };
        idx.iter()
            .map(|x| {
                x.iter()
                    .map(|y| {
                        let bound: usize = 1 << (self.config.lg_n - self.config.rate_bits);
                        if (y % n) > bound && round == 0 {
                            return 0;
                        };
                        let y = if round == 0 && self.config.transposed_input {
                            let row = y / (1 << self.config.lg_n);
                            let col = y % (1 << self.config.lg_n);
                            col * self.config.k + row
                        } else {
                            *y
                        };

                        y * SIZE_F * self.config.extension + addr_input
                    })
                    .collect()
            })
            .collect()
    }

    fn idx_to_addro(&self, idx: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        idx.iter()
            .map(|x| {
                x.iter()
                    .map(|y| y * SIZE_F * self.config.extension + self.config.addr_output)
                    .collect()
            })
            .collect()
    }

    fn idx_to_addr(&self, idx: Vec<Vec<usize>>, offset: bool, round: usize) -> Vec<Vec<usize>> {
        let n = 1 << self.config.lg_n;
        let addr_input = if round == 0 {
            self.config.addr_input
        } else {
            self.config.addr_tmp
        };
        idx.iter()
            .map(|x| {
                x.iter()
                    .map(|y| {
                        let bound: usize = 1 << (self.config.lg_n - self.config.rate_bits);
                        if (y % n) > bound && round == 0 && !offset {
                            return 0;
                        };
                        let y: usize = if round == 0 && !offset && self.config.transposed_input {
                            let row = y / (1 << self.config.lg_n);
                            let col = y % (1 << self.config.lg_n);
                            col * self.config.k + row
                        } else {
                            *y
                        };
                        y * SIZE_F * self.config.extension
                            + if offset {
                                self.config.addr_output
                            } else {
                                addr_input
                            }
                    })
                    .collect()
            })
            .collect()
    }

    fn idx_bit_reverse(&self, idx: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
        let n = 1 << self.config.lg_n;
        idx.iter()
            .map(|x| {
                x.iter()
                    .map(|y| *y / n * n + bit_reverse(*y % n, self.config.lg_n))
                    .collect()
            })
            .collect()
    }

    fn plane_offset(&self, now_dim: u32, factor: usize) -> usize {
        self.config.get_plane_length().pow(now_dim) / factor
    }
}

pub fn fft(input: &Vec<F>, inverse: bool, reverse: bool) -> Vec<F> {
    let length = input.len();
    let mut a = input.clone();
    let mut bit = 0;
    while (1 << bit) < length {
        bit += 1;
    }
    let mut rev = vec![0; length];
    for i in 0..length {
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        if i < rev[i] {
            let tmp = a[rev[i]];
            a[rev[i]] = a[i];
            a[i] = tmp;
        }
    }
    let mut mid = 1;
    while mid < length {
        let tmp = F::primitive_root_of_unity(log2_strict(mid * 2));
        let tw = if inverse {
            tmp.try_inverse().unwrap()
        } else {
            tmp
        };
        for i in (0..length).step_by(mid * 2) {
            let mut omega: F = F::ONE;
            for j in 0..mid {
                let x = a[i + j];
                let y = omega * a[i + j + mid];
                a[i + j] = x + y;
                a[i + j + mid] = x - y;
                omega = omega * tw;
            }
        }
        mid = mid * 2;
    }
    if inverse {
        let inv = F::from_canonical_u64(length as u64).try_inverse().unwrap();
        for i in 0..length {
            a[i] = a[i] * inv;
        }
    }
    if reverse {
        let mut rev = vec![0; length];
        for i in 0..length {
            rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
            if i < rev[i] {
                let tmp = a[rev[i]];
                a[rev[i]] = a[i];
                a[i] = tmp;
            }
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use crate::kernel::fft::{fft, Fft, FftConfig, FftDirection};
    use plonky2::field::goldilocks_field::GoldilocksField as F;
    use plonky2::field::types::Field;

    #[test]
    fn test_create_prefetch() {
        let config = FftConfig {
            lg_n: 3,
            k: 20,
            direction: FftDirection::NN,
            inverse: false,
            addr_input: 0,
            addr_tmp: 0,
            addr_output: 1 << 10,
            rate_bits: 0,
            coset: false,
            extension: 1,
            transposed_input: false,
        };
        let _fft = Fft::new(config);
    }

    #[test]
    fn test_fft() {
        const LENGTH: usize = 8;
        let input = (0..LENGTH)
            .map(|x| F::from_canonical_u64(x as u64))
            .collect();
        let output_fft_expected: Vec<u64> = vec![
            28,
            18445622567621360637,
            18445618169507741693,
            1130298020461564,
            18446744069414584317,
            18445613771394122749,
            1125899906842620,
            1121501793223676,
        ];
        let output_fft_expected = output_fft_expected
            .into_iter()
            .map(|x| F::from_canonical_u64(x))
            .collect::<Vec<_>>();

        let output_fft = fft(&input, false, false);
        let output_ifft = fft(&output_fft, true, false);

        assert_eq!(output_fft, output_fft_expected, "fft result does not match");
        assert_eq!(output_ifft, input, "ifft result does not match");
    }
}
