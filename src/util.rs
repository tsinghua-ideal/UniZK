use crate::config::{arch_config::ARCH_CONFIG, enable_config::ENABLE_CONFIG};
use clap::{value_parser, Arg, Command};
use log::info;

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

pub fn set_config() {
    let args = Command::new("simulator_v2")
        .version("1.0")
        .about("Demonstrates command line argument parsing")
        .arg(
            Arg::new("ram")
                .short('r')
                .long("ram")
                .default_value("8")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("tiles")
                .short('t')
                .long("tiles")
                .default_value("32")
                .value_parser(value_parser!(usize)),
        )
        .arg(
            Arg::new("enable")
                .short('e')
                .long("enable")
                .default_value("-1")
                .value_parser(value_parser!(i32)),
        )
        .get_matches();
    let ram_size: &usize = args.get_one::<usize>("ram").unwrap();
    let tiles: &usize = args.get_one::<usize>("tiles").unwrap();
    let enable: &i32 = args.get_one::<i32>("enable").unwrap();

    unsafe {
        ARCH_CONFIG.rdbuf_sz_kb = ram_size * 1024 / 2;
        ARCH_CONFIG.wrbuf_sz_kb = ram_size * 1024 / 2;
        ARCH_CONFIG.num_tiles = *tiles;

        if *enable >= 0 {
            ENABLE_CONFIG.fft = false;
            ENABLE_CONFIG.hash = false;
            ENABLE_CONFIG.other = false;
            match enable {
                0 => {
                    ENABLE_CONFIG.fft = true;
                }
                1 => {
                    ENABLE_CONFIG.hash = true;
                }
                _ => {
                    panic!("Invalid enable option")
                }
            }
        }

        println!("ARCH_CONFIG: {:?}", ARCH_CONFIG);
    }
}