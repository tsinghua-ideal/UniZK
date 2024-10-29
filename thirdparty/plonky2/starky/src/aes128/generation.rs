#![allow(clippy::needless_range_loop)]

use core::convert::TryInto;
use core::usize;

use arrayref::{array_mut_ref, array_ref};
use plonky2::field::polynomial::PolynomialValues;
use plonky2::field::types::Field;
use plonky2::iop::target;

use super::constants::*;
use super::key_u8_bit;
use super::layout::*;
use crate::util::trace_rows_to_poly_values;

type StateType = [u8; 16];
type KeyType = [u32; 4];

fn is_power_of_two(n: u64) -> bool {
    n & (n - 1) == 0
}

#[repr(transparent)]
pub struct AesTrace<F: Field>(Vec<[F; NUM_COLS]>);

impl<F: Field> AesTrace<F> {
    pub fn new(max_rows: usize) -> AesTrace<F> {
        assert!(
            is_power_of_two(max_rows as u64),
            "max_rows must be a power of two"
        );
        AesTrace(vec![[F::ZERO; NUM_COLS]; max_rows])
    }
}

pub struct AesTraceGenerator<F: Field> {
    trace: AesTrace<F>,
}

impl<F: Field> AesTraceGenerator<F> {
    pub fn new(max_rows: usize) -> AesTraceGenerator<F> {
        AesTraceGenerator {
            trace: AesTrace::new(max_rows),
        }
    }

    fn max_rows(&self) -> usize {
        self.trace.0.len()
    }

    fn get_spec_window(&mut self, step: usize) -> (&mut [[F; NUM_COLS]; 2], usize) {
        let idx = step;
        assert!(idx >= 0, "step must be >= 0");
        assert!(idx < self.max_rows(), "get_curr_window exceeded MAX_ROWS");
        (array_mut_ref![self.trace.0, idx, 2], idx)
    }

    fn get_spec_row(&mut self, step: usize) -> &mut [F; NUM_COLS] {
        let idx = step;
        assert!(idx >= 0, "step must be >= 0");
        assert!(idx < self.max_rows(), "get_curr_window exceeded MAX_ROWS");
        &mut self.trace.0[idx]
    }

    fn copy_pt(&mut self, pt: StateType) {
        let curr_row = self.get_spec_row(0);
        for i in 0..16 {
            for j in 0..8 {
                curr_row[input_i_bit(i, j)] = F::from_canonical_u32(((pt[i] >> j) & 1) as u32);
            }
        }
    }

    fn keep_state(&mut self, step: usize) {
        let ([curr_row, next_row], _) = self.get_spec_window(step);
        for i in 0..16 {
            for j in 0..8 {
                next_row[input_i_bit(i, j)] = curr_row[input_i_bit(i, j)];
            }
        }
    }

    fn copy_key(&mut self, key: KeyType) {
        let curr_row = self.get_spec_row(0);
        for i in 0..4 {
            for j in 0..32 {
                curr_row[key_i_bit(i, j)] = F::from_canonical_u32(((key[i] >> j) & 1) as u32);
            }
        }
    }

    fn keep_key(&mut self, step: usize) {
        let ([curr_row, next_row], _) = self.get_spec_window(step);
        for i in 0..4 {
            for j in 0..32 {
                next_row[key_i_bit(i, j)] = curr_row[key_i_bit(i, j)];
            }
        }
    }

    fn gen_add_round_key(&mut self, step: usize, state: StateType, key: KeyType) -> StateType {
        let ([curr_row, next_row], _) = self.get_spec_window(step);
        let mut new_state = [0; 16];
        for i in 0..4 {
            let rk = key[i].to_be_bytes();
            for j in 0..4 {
                new_state[4 * i + j] = state[4 * i + j] ^ rk[j];
            }
        }

        for i in 0..16 {
            for j in 0..8 {
                next_row[input_i_bit(i, j)] =
                    F::from_canonical_u32(((new_state[i] >> j) & 1) as u32);
            }
        }

        new_state
    }

    fn gen_sub(
        &mut self,
        start_step: usize,
        a: u8,
        sub_bit_start: usize,
        target_next_row: bool,
        target_bit_start: usize,
    ) -> u8 {
        let r0 = 0x11b;
        let b = (eea_v2(r0, a as i32) & 0xFF) as u8;

        let gmul_start = sub_bit_start + NUM_INV_BITS;
        let mut p = 0;
        let mut temp_a = a;
        let mut temp_b = b;
        for i in 0..8 {
            let ([curr_row, next_row], _) = self.get_spec_window(start_step + i);

            // inv
            for bit in 0..8 {
                curr_row[sub_bit_start + bit] = F::from_canonical_u32(((b >> bit) & 1) as u32);
            }

            // a * bi
            let temp_a_bi = if temp_b & 1 == 1 { temp_a } else { 0 };
            for bit in 0..8 {
                curr_row[key_gmul_a_bi_bit(gmul_start, bit)] =
                    F::from_canonical_u32(((temp_a_bi >> bit) & 1) as u32);
            }

            p ^= temp_a_bi;
            for bit in 0..8 {
                curr_row[key_gmul_p_bit(gmul_start, bit)] =
                    F::from_canonical_u32(((p >> bit) & 1) as u32);
            }

            let hi_bit_set = (temp_a & 0x80) != 0;
            temp_a <<= 1;
            let temp_a_1b = temp_a ^ 0x1b;
            for bit in 0..8 {
                curr_row[key_gmul_a_shift_xor_1b_bit(gmul_start, bit)] =
                    F::from_canonical_u32(((temp_a_1b >> bit) & 1) as u32);
            }
            if hi_bit_set {
                temp_a = temp_a_1b;
            }
            for bit in 0..8 {
                curr_row[key_gmul_a_tmp_bit(gmul_start, bit)] =
                    F::from_canonical_u32(((temp_a >> bit) & 1) as u32);
            }

            temp_b >>= 1;
        }

        let ([curr_row, next_row], _) = self.get_spec_window(start_step + 7);
        let offset = [0, 4, 5, 6, 7];
        let mut res: Vec<u8> = (0..8).map(|i| ((b >> i) & 1)).collect();
        for i in 0..8 {
            for j in 1..5 {
                res[i] = res[i] ^ ((b >> ((i + offset[j]) % 8)) & 1);
                curr_row[sbox_xor_bit(sub_bit_start, j - 1, i)] =
                    F::from_canonical_u32(res[i] as u32);
            }
            res[i] = res[i] ^ BITS99LSB[i];

            if target_next_row {
                next_row[target_bit_start + i] = F::from_canonical_u32(res[i] as u32);
            } else {
                curr_row[target_bit_start + i] = F::from_canonical_u32(res[i] as u32);
            }
        }

        (0..8).map(|i| res[i] << i).fold(0, |acc, x| acc | x)
    }

    fn gen_key_update(&mut self, step: usize, round: usize, key: KeyType) -> KeyType {
        let key_3_rot = rot_word(key[3]);

        let key_3_0 = (key_3_rot >> 24) as u8;
        let key_3_1 = (key_3_rot >> 16) as u8;
        let key_3_2 = (key_3_rot >> 8) as u8;
        let key_3_3 = key_3_rot as u8;

        let key_3_0 = self.gen_sub(step, key_3_0, key_3_0_bit(0), false, KEY_3_SUB + 24);
        let key_3_1 = self.gen_sub(step, key_3_1, key_3_1_bit(0), false, KEY_3_SUB + 16);
        let key_3_2 = self.gen_sub(step, key_3_2, key_3_2_bit(0), false, KEY_3_SUB + 8);
        let key_3_3 = self.gen_sub(step, key_3_3, key_3_3_bit(0), false, KEY_3_SUB);

        // keep state
        // state in step + 0 plus key should already be in step + 1
        for i in 1..8 {
            self.keep_state(step + i);
        }

        // keep key
        for i in 0..7 {
            self.keep_key(step + i);
        }

        let key_3 = (key_3_0 as u32) << 24
            | (key_3_1 as u32) << 16
            | (key_3_2 as u32) << 8
            | (key_3_3 as u32);
        let key_3 = key_3 ^ RCON[round];
        let ([curr_row, next_row], _) = self.get_spec_window(step + 7);
        for i in 0..8 {
            curr_row[key_3_rconv_bit(i)] = F::from_canonical_u32(((key_3 >> 24 >> i) & 1) as u32);
        }

        let mut new_key = [0; 4];
        new_key[0] = key[0] ^ key_3;
        new_key[1] = key[1] ^ new_key[0];
        new_key[2] = key[2] ^ new_key[1];
        new_key[3] = key[3] ^ new_key[2];

        for i in 0..32 {
            next_row[key_i_bit(0, i)] = F::from_canonical_u32(((new_key[0] >> i) & 1) as u32);
            next_row[key_i_bit(1, i)] = F::from_canonical_u32(((new_key[1] >> i) & 1) as u32);
            next_row[key_i_bit(2, i)] = F::from_canonical_u32(((new_key[2] >> i) & 1) as u32);
            next_row[key_i_bit(3, i)] = F::from_canonical_u32(((new_key[3] >> i) & 1) as u32);
        }

        new_key
    }

    fn gen_rcon(&mut self, step: usize, round: usize) {
        let ([curr_row, _], _) = self.get_spec_window(step);
        for i in 0..8 {
            curr_row[RCON_START + i] = F::from_canonical_u32((RCON[round] >> 24 >> i) & 1);
        }
    }

    fn gen_state_sbox(&mut self, step: usize, state: StateType) -> StateType {
        let mut new_state = [0; 16];
        for i in 0..16 {
            let res = self.gen_sub(
                step,
                state[i],
                state_i_sub_start(i),
                true,
                input_i_bit(i, 0),
            );
            new_state[i] = res;
        }

        // keep key
        for i in 0..8 {
            self.keep_key(step + i);
        }

        new_state
    }

    fn gen_shift_rows(&mut self, step: usize, state: StateType) -> StateType {
        self.keep_key(step);

        let mut new_state = [0; 16];
        new_state[0] = state[0];
        new_state[4] = state[4];
        new_state[8] = state[8];
        new_state[12] = state[12];

        new_state[1] = state[5];
        new_state[5] = state[9];
        new_state[9] = state[13];
        new_state[13] = state[1];

        new_state[2] = state[10];
        new_state[10] = state[2];
        new_state[6] = state[14];
        new_state[14] = state[6];

        new_state[3] = state[15];
        new_state[15] = state[11];
        new_state[11] = state[7];
        new_state[7] = state[3];

        let ([curr_row, next_row], _) = self.get_spec_window(step);
        for i in 0..16 {
            for j in 0..8 {
                next_row[input_i_bit(i, j)] =
                    F::from_canonical_u32(((new_state[i] >> j) & 1) as u32);
            }
        }

        new_state
    }

    fn gen_mix_columns(&mut self, step: usize, state: StateType) -> StateType {
        // keep key
        for i in 0..8 {
            self.keep_key(step + i);
        }

        // keep state
        for i in 0..7 {
            self.keep_state(step + i);
        }

        let mut state_gmul_tmp = [[0; 2]; 16];
        let mut state_xor_tmp = [[0; 2]; 16];
        let mut new_state = [0; 16];

        for i in 0..4 {
            let s = (0..4).map(|j| state[4 * i + j]).collect::<Vec<_>>();

            for j in 0..4 {
                for k in 0..2 {
                    let constant = if k == 0 { 2u8 } else { 3u8 };
                    let gmul_start = if k == 0 {
                        state_i_gmul_2_start(i * 4 + j)
                    } else {
                        state_i_gmul_3_start(i * 4 + j)
                    };

                    // gmul const
                    let mut p = 0;
                    let mut temp_a = constant;
                    let mut temp_b = s[j];
                    for substep in step..step + NUM_GMUL_STEPS {
                        let ([curr_row, next_row], _) = self.get_spec_window(substep);

                        // a * bi
                        let temp_a_bi = if temp_b & 1 == 1 { temp_a } else { 0 };
                        for bit in 0..8 {
                            curr_row[state_gmul_const_a_bi_bit(gmul_start, bit)] =
                                F::from_canonical_u32(((temp_a_bi >> bit) & 1) as u32);
                        }

                        // p
                        p ^= temp_a_bi;
                        for bit in 0..8 {
                            curr_row[state_gmul_const_p_bit(gmul_start, bit)] =
                                F::from_canonical_u32(((p >> bit) & 1) as u32);
                        }

                        let hi_bit_set = (temp_a & 0x80) != 0;
                        temp_a <<= 1;
                        let temp_a_1b = temp_a ^ 0x1b;
                        if hi_bit_set {
                            temp_a = temp_a_1b;
                        }

                        temp_b >>= 1;
                    }

                    state_gmul_tmp[4 * i + j][k] = p;
                }
            }

            state_xor_tmp[4 * i][0] = state_gmul_tmp[4 * i][0] ^ state_gmul_tmp[4 * i + 1][1];
            state_xor_tmp[4 * i][1] = state_xor_tmp[4 * i][0] ^ s[2];
            new_state[4 * i] = state_xor_tmp[4 * i][1] ^ s[3];

            state_xor_tmp[4 * i + 1][0] = s[0] ^ state_gmul_tmp[4 * i + 1][0];
            state_xor_tmp[4 * i + 1][1] =
                state_xor_tmp[4 * i + 1][0] ^ state_gmul_tmp[4 * i + 2][1];
            new_state[4 * i + 1] = state_xor_tmp[4 * i + 1][1] ^ s[3];

            state_xor_tmp[4 * i + 2][0] = s[0] ^ s[1];
            state_xor_tmp[4 * i + 2][1] =
                state_xor_tmp[4 * i + 2][0] ^ state_gmul_tmp[4 * i + 2][0];
            new_state[4 * i + 2] = state_xor_tmp[4 * i + 2][1] ^ state_gmul_tmp[4 * i + 3][1];

            state_xor_tmp[4 * i + 3][0] = state_gmul_tmp[4 * i][1] ^ s[1];
            state_xor_tmp[4 * i + 3][1] = state_xor_tmp[4 * i + 3][0] ^ s[2];
            new_state[4 * i + 3] = state_xor_tmp[4 * i + 3][1] ^ state_gmul_tmp[4 * i + 3][0];

            // write xor tmp
            let ([curr_row, next_row], _) = self.get_spec_window(step + NUM_GMUL_STEPS - 1);
            for j in 0..4 {
                for k in 0..8 {
                    curr_row[state_i_xor_start(4 * i + j, 0) + k] =
                        F::from_canonical_u32(((state_xor_tmp[4 * i + j][0] >> k) & 1) as u32);
                    curr_row[state_i_xor_start(4 * i + j, 1) + k] =
                        F::from_canonical_u32(((state_xor_tmp[4 * i + j][1] >> k) & 1) as u32);
                }
            }

            // write state to next row
            for j in 0..4 {
                for k in 0..8 {
                    next_row[input_i_bit(4 * i + j, k)] =
                        F::from_canonical_u32(((new_state[4 * i + j] >> k) & 1) as u32);
                }
            }
        }

        new_state
    }

    pub fn gen_aes(&mut self, pt: StateType, key: [u8; 16]) -> StateType {
        let state = pt;
        let mut key32 = [0; 4];
        let mut round = 0;
        let mut step = 0;
        for i in 0..4 {
            key32[i] =
                u32::from_be_bytes([key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]]);
        }
        let key = key32;

        // copy pt to trace
        self.copy_pt(state);

        // copy key to trace
        self.copy_key(key);

        // add round key
        let mut state = self.gen_add_round_key(step, state, key);
        let mut key = self.gen_key_update(step, round, key);
        for substep in step..step + NUM_KEY_UPDATE_STEPS {
            self.gen_rcon(substep, round);
        }
        round += 1;
        step += NUM_KEY_UPDATE_STEPS;

        for _ in 1..10 {
            // println!("state sbox step: {}", step);
            state = self.gen_state_sbox(step, state);
            for substep in step..step + NUM_STATE_SBOX_STEPS {
                self.gen_rcon(substep, round);
            }
            step += NUM_STATE_SBOX_STEPS;

            // println!("shift rows step: {}", step);
            state = self.gen_shift_rows(step, state);
            self.gen_rcon(step, round);
            step += 1;

            state = self.gen_mix_columns(step, state);
            for substep in step..step + NUM_MIX_COL_STEPS {
                self.gen_rcon(substep, round);
            }
            step += NUM_MIX_COL_STEPS;

            // println!("add round key step: {}", step);
            state = self.gen_add_round_key(step, state, key);
            key = self.gen_key_update(step, round, key);
            for substep in step..step + NUM_KEY_UPDATE_STEPS {
                self.gen_rcon(substep, round);
            }
            step += NUM_KEY_UPDATE_STEPS;
            // println!("Round {} finished! Total steps: {}", round, step);
            round += 1;
        }
        state = self.gen_state_sbox(step, state);
        for substep in step..step + NUM_STATE_SBOX_STEPS {
            self.gen_rcon(substep, round);
        }
        step += NUM_STATE_SBOX_STEPS;

        state = self.gen_shift_rows(step, state);
        self.gen_rcon(step, round);
        step += 1;

        state = self.gen_add_round_key(step, state, key);
        self.keep_key(step);
        for substep in step..step + 1 {
            self.gen_rcon(substep, round);
        }
        step += 1;

        // step bit
        for i in 0..step + 1 {
            let ([curr_row, _], _) = self.get_spec_window(i);
            for j in 0..step + 1 {
                curr_row[step_bit(j)] = F::from_canonical_u32((j == i) as u32);
            }
        }

        println!("Generation finished! Total steps: {}", step);

        //////////////
        // Check
        // for i in 0..36 {
        //     let ([curr_row, next_row], _) = self.get_spec_window(i);
        //     let s = (0..16)
        //         .map(|j| {
        //             (0..8)
        //                 .map(|k| curr_row[input_i_bit(j, k)] * F::from_canonical_u32(1 << k))
        //                 .sum::<F>()
        //         })
        //         .collect::<Vec<_>>();
        //     println!("state = {:02x?} in step {}", s, i);
        // }
        // for i in 0..NUM_STEPS {
        //     let ([curr_row, next_row], _) = self.get_spec_window(i);
        //     let s = (0..16)
        //         .map(|j| {
        //             (0..8)
        //                 .map(|k| curr_row[key_u8_bit(j, k)] * F::from_canonical_u32(1 << k))
        //                 .sum::<F>()
        //         })
        //         .collect::<Vec<_>>();
        //     println!("key = {:02x?} in step {}", s, i);
        // }

        // for i in 0..NUM_STEPS - 1 {
        //     let ([curr_row, next_row], _) = self.get_spec_window(i);
        //     for bit in 0..NUM_STEPS {
        //         assert!(
        //             next_row[step_bit((bit + 1) % NUM_STEPS)] == curr_row[step_bit(bit)],
        //             "step_bit transition failed at step {}",
        //             i
        //         )
        //     }
        // }

        // let mut start_row = 0;
        // for i in 0..10 {
        //     for step in start_row..ROUND_CHANGE_STEPS[i] + 1 {
        //         let ([curr_row, next_row], _) = self.get_spec_window(step);
        //         let rc = (0..8)
        //             .map(|j| curr_row[RCON_START + j] * F::from_canonical_u32((1 << j) as u32))
        //             .sum::<F>();
        //         assert!(
        //             rc == F::from_canonical_u32((RCON[i] >> 24) as u32),
        //             "RCON transition failed at step {}",
        //             step
        //         );
        //     }
        //     start_row = ROUND_CHANGE_STEPS[i] + 1;
        // }

        // for round in 0..11 {
        //     let ([curr_row, next_row], _) = self.get_spec_window(ADD_ROUND_KEY_STEPS[round]);
        //     for i in 0..16 {
        //         let s = (0..8)
        //             .map(|j| curr_row[input_i_bit(i, j)] * F::from_canonical_u32(1 << j))
        //             .sum::<F>();
        //         let k = (0..8)
        //             .map(|j| curr_row[key_u8_bit(i, j)] * F::from_canonical_u32(1 << j))
        //             .sum::<F>();
        //         let new_s = (0..8)
        //             .map(|j| next_row[input_i_bit(i, j)] * F::from_canonical_u32(1 << j))
        //             .sum::<F>();
        //         for bit in 0..8 {
        //             let computed_bit = curr_row[input_i_bit(i, bit)] + curr_row[key_u8_bit(i, bit)]
        //                 - curr_row[input_i_bit(i, bit)] * curr_row[key_u8_bit(i, bit)] * F::TWO;
        //             assert!(
        //                 computed_bit == next_row[input_i_bit(i, bit)],
        //                 "add round key transition failed at step {}",
        //                 ADD_ROUND_KEY_STEPS[round]
        //             );
        //         }
        //     }
        // }

        // for step in 0..NUM_STEPS {
        //     let mut key_same = true;
        //     for i in 0..4 {
        //         for bit in 0..32 {
        //             let ([curr_row, next_row], _) = self.get_spec_window(step);
        //             if curr_row[key_i_bit(i, bit)] != next_row[key_i_bit(i, bit)] {
        //                 key_same = false;
        //                 break;
        //             }
        //         }
        //     }
        //     if !key_same{
        //         println!("Key update failed at step {}", step);
        //     }
        // }

        state
    }

    pub fn into_polynomial_values(self) -> Vec<PolynomialValues<F>> {
        trace_rows_to_poly_values(self.trace.0)
    }
}

pub fn to_u32_array_be<const N: usize>(block: [u8; N * 4]) -> [u32; N] {
    let mut block_u32 = [0; N];
    for (o, chunk) in block_u32.iter_mut().zip(block.chunks_exact(4)) {
        *o = u32::from_be_bytes(chunk.try_into().unwrap());
    }
    block_u32
}

fn rot_word(word: u32) -> u32 {
    (word << 8) | (word >> 24)
}

// 获取最高位
fn get_highest_position(mut number: u16) -> i32 {
    let mut i = 0;
    while number != 0 {
        i += 1;
        number >>= 1;
    }
    i
}

// GF(2^8)的多项式除法
fn division(num_l: u16, num_r: u16) -> u8 {
    let mut r0 = num_l;
    let mut q = 0u8;
    let mut bit_count = get_highest_position(r0) - get_highest_position(num_r);

    while bit_count >= 0 {
        q |= 1 << bit_count;
        r0 ^= num_r << bit_count;
        bit_count = get_highest_position(r0) - get_highest_position(num_r);
    }

    q
}

// GF(2^8)多项式乘法
fn multiplication(num_l: u8, num_r: u8) -> u16 {
    let mut result = 0u16;

    for i in 0..8 {
        result ^= u16::from((num_l >> i) & 0x01) * (u16::from(num_r) << i);
    }

    result
}

// 伽罗瓦域的扩展欧几里得算法
fn eea_v2(mut r0: i32, mut r1: i32) -> i32 {
    let mod_value = r0;
    let mut t0 = 0;
    let mut t1 = 1;
    let mut t = t1;

    if r1 == 0 {
        return 0;
    }

    while r1 != 1 {
        let q = division(r0 as u16, r1 as u16);

        let r = r0 ^ (multiplication(q, r1 as u8) as i32);
        t = t0 ^ (multiplication(q, t1 as u8) as i32);

        r0 = r1;
        r1 = r;
        t0 = t1;
        t1 = t;
    }

    if t < 0 {
        t ^= mod_value;
    }

    t
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2_util::log2_ceil;

    use super::super::aes_impl::AESwcs;
    use super::*;

    type F = GoldilocksField;

    #[test]
    fn test_zero_key_and_zero_input() {
        let key = [0u8; 16];
        let input = [0u8; 16];

        let mut aes = AESwcs::new(&key);
        let output = aes.encrypt(&input);
        println!("AES output: {:02x?}", output);

        let max_rows = 1 << log2_ceil(NUM_STEPS);
        let mut aes_trace = AesTraceGenerator::<F>::new(max_rows);
        let output = aes_trace.gen_aes(input, key);
        println!("AES trace output: {:02x?}", output);
    }

    // #[test]
    // fn test_hash_of_zero() {

    //     let block = [0u8; 64];
    //     let block_arr = GenericArray::<u8, U64>::from(block);
    //     let mut state = HASH_IV;
    //     compress256(&mut state, &[block_arr]);

    //     let left_input = [0u32; 8];
    //     let right_input = [0u32; 8];
    //     let mut generator = AesTraceGenerator::<F>::new(128);

    //     let his = generator.gen_hash(left_input, right_input);

    //     assert_eq!(his, state);
    // }

    // #[test]
    // fn test_hash_of_something() {
    //     let mut block = [0u8; 64];
    //     for i in 0..64 {
    //         block[i] = i as u8;
    //     }

    //     let block_arr = GenericArray::<u8, U64>::from(block);
    //     let mut state = HASH_IV;
    //     compress256(&mut state, &[block_arr]);

    //     let block: [u32; 16] = to_u32_array_be(block);
    //     let left_input = *array_ref![block, 0, 8];
    //     let right_input = *array_ref![block, 8, 8];
    //     let mut generator = AesTraceGenerator::<F>::new(128);

    //     let his = generator.gen_hash(left_input, right_input);
    //     assert_eq!(his, state);
    // }
}
