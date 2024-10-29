pub const SIZE_F: usize = 8;
pub const SPONGE_RATE: usize = 8;
pub const SPONGE_CAPACITY: usize = 4;
pub const SPONGE_WIDTH: usize = SPONGE_RATE + SPONGE_CAPACITY;
pub const NUM_HASH_OUT_ELTS: usize = 4;
pub const SALT_SIZE: usize = 4;

pub const FRI_PROOF_OF_WORK_ROUND: usize = 200000;

pub const D: usize = 2; // for extension field
pub const B: usize = 2; // for base sum gate
pub const SINGLE_HASH_DELAY: usize = 24 * 4 + 14 + 130 * 6 + 24 * 4;

pub const BUFSIZE: usize = 8; // for bin trace file writing

pub const BATCH_SIZE: usize = 256;

pub const HASH_COMPUTATION: usize = (4 + 12) * 12 * 8 + 12 * 12 + 27 * 22;

pub type AddrTrans = fn(usize) -> usize;

pub fn default_addr_trans(addr: usize) -> usize {
    addr
}

pub fn ceil_div_usize(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

pub fn bit_reverse(mut x: usize, n: usize) -> usize {
    let mut y = 0;
    for _i in 0..n {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    y
}

pub fn log2(x: usize) -> usize {
    let mut y = 0;
    let mut z = x;
    while z != 0 {
        z >>= 1;
        y += 1;
    }
    y - 1
}
