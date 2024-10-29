use super::layout::NUM_STEPS;
// for 10 times key update, extra 0x00000000 element for padding
pub const RCON: [u32; 11] = [
    0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b000000, 0x36000000, 0x00000000,
];

pub const BITS99LSB: [u8; 8] = [1, 1, 0, 0, 0, 1, 1, 0];

pub const ROUND_CHANGE_STEPS: [usize; 11] = [7, 32, 57, 82, 107, 132, 157, 182, 207, 232, 250];
pub const ADD_ROUND_KEY_STEPS: [usize; 11] = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 242];

pub const KEY_UPDATE_STEPS: [usize; 10] = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225];
pub const STATE_SBOX_STEPS: [usize; 10] = [8, 33, 58, 83, 108, 133, 158, 183, 208, 233];
pub const STATE_SHIFT_ROWS_STEPS: [usize; 10] = [16, 41, 66, 91, 116, 141, 166, 191, 216, 241];
pub const STATE_MIX_COLUMNS_STEPS: [usize; 9] = [17, 42, 67, 92, 117, 142, 167, 192, 217];

pub const SHIFT_ROWS: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
