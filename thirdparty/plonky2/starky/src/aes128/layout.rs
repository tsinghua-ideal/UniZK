pub const NUM_PIS: usize = 0;
pub const NUM_COLS: usize = LAST_COL + 1;
pub const STEP_BITS_START: usize = 0;
pub const NUM_STEPS: usize = 244;

pub const NUM_GMUL_STEPS: usize = 8;
pub const NUM_KEY_UPDATE_STEPS: usize = NUM_GMUL_STEPS;
pub const NUM_STATE_SBOX_STEPS: usize = NUM_GMUL_STEPS;
pub const NUM_MIX_COL_STEPS: usize = NUM_GMUL_STEPS;

pub fn step_bit(i: usize) -> usize {
    STEP_BITS_START + i
}

pub const RCON_START: usize = STEP_BITS_START + NUM_STEPS;

pub const STATE_START: usize = RCON_START + 8;
pub fn input_i_bit(i: usize, j: usize) -> usize {
    assert!(i < 16);
    assert!(j < 8);
    STATE_START + i * 8 + j
}

pub const KEY_START: usize = STATE_START + 128;
pub fn key_i_bit(i: usize, j: usize) -> usize {
    assert!(i < 4);
    assert!(j < 32);
    KEY_START + i * 32 + j
}

// the number of bits in the S-box of a single byte
// inv 8-bit + gmul + final xor 32-bit
pub const NUM_SUB_BITS: usize = 8 + NUM_GMUL_BITS + 32;

pub const KEY_3_ROT: usize = KEY_START + 128;
pub fn key_3_0_bit(i: usize) -> usize {
    assert!(i < NUM_SUB_BITS);
    KEY_3_ROT + i
}
pub fn key_3_1_bit(i: usize) -> usize {
    assert!(i < NUM_SUB_BITS);
    KEY_3_ROT + NUM_SUB_BITS + i
}
pub fn key_3_2_bit(i: usize) -> usize {
    assert!(i < NUM_SUB_BITS);
    KEY_3_ROT + 2 * NUM_SUB_BITS + i
}
pub fn key_3_3_bit(i: usize) -> usize {
    assert!(i < NUM_SUB_BITS);
    KEY_3_ROT + 3 * NUM_SUB_BITS + i
}

pub const NUM_INV_BITS: usize = 8;

pub const NUM_GMUL_BITS: usize = 32;

pub fn key_gmul_a_bi_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + i
}
pub fn key_gmul_p_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + 8 + i
}
pub fn key_gmul_a_shift_xor_1b_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + 16 + i
}
pub fn key_gmul_a_tmp_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + 24 + i
}

pub fn sbox_xor_bit(start: usize, i: usize, j: usize) -> usize {
    start + 8 + NUM_GMUL_BITS + i * 8 + j
}

pub const KEY_3_SUB: usize = KEY_3_ROT + 4 * NUM_SUB_BITS;
pub fn key_3_sub_bit(i: usize, j: usize) -> usize {
    assert!(i < 4);
    assert!(j < 8);
    KEY_3_SUB + i * 8 + j
}

pub const KEY_3_RCONV: usize = KEY_3_SUB + 32;
pub fn key_3_rconv_bit(i: usize) -> usize {
    assert!(i < 8);
    KEY_3_RCONV + i
}

pub const STATE_SUB: usize = KEY_3_RCONV + 8;
pub fn state_i_sub_start(i: usize) -> usize {
    STATE_SUB + i * NUM_SUB_BITS
}

pub const STATE_GMUL: usize = STATE_SUB + 16 * NUM_SUB_BITS;
pub const NUM_GMUL_CONST_BITS: usize = 16;
pub fn state_i_gmul_2_start(i: usize) -> usize {
    STATE_GMUL + i * 2 * NUM_GMUL_CONST_BITS
}
pub fn state_i_gmul_3_start(i: usize) -> usize {
    STATE_GMUL + (i * 2 + 1) * NUM_GMUL_CONST_BITS
}

pub fn state_gmul_const_a_bi_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + i
}
pub fn state_gmul_const_p_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + 8 + i
}
pub fn state_gmul_const_res_bit(start: usize, i: usize) -> usize {
    assert!(i < 8);
    start + 8 + i
}

pub const STATE_XOR: usize = STATE_GMUL + 16 * 2 * NUM_GMUL_CONST_BITS;
// bit start of res of j-th xor op of i-th state
pub fn state_i_xor_start(i: usize, j: usize) -> usize {
    STATE_XOR + i * 2 * 8 + j * 8
}

pub const LAST_COL: usize = STATE_XOR + 16 * 8 * 2;
