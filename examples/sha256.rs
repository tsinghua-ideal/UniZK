use env_logger::Env;
use log::info;

use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use plonky2_sha256::circuit::{array_to_bits, make_circuits};

use sha2::{Digest, Sha256};
use unizk::config::arch_config::ARCH_CONFIG;
use unizk::config::enable_config::ENABLE_CONFIG;
use unizk::config::RamConfig;
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;

use clap::{value_parser, Arg, Command};
fn main() {
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

    // unsafe {
    //     ARCH_CONFIG.rdbuf_sz_kb = ram_size / 2 * 1024;
    //     ARCH_CONFIG.wrbuf_sz_kb = ram_size / 2 * 1024;
    //     ARCH_CONFIG.num_tiles = *tiles;
    // }

    unsafe {
        if *enable >= 0 {
            ENABLE_CONFIG.fft = false;
            ENABLE_CONFIG.transpose = false;
            ENABLE_CONFIG.tree = false;
            ENABLE_CONFIG.poly = false;
            ENABLE_CONFIG.hash = false;
            match enable {
                0 => {
                    ENABLE_CONFIG.fft = true;
                }
                1 => {
                    ENABLE_CONFIG.tree = true;
                }
                2 => {
                    ENABLE_CONFIG.poly = true;
                }
                _ => {
                    panic!("Invalid enable option")
                }
            }
        }
    }
    let kernel_name = match enable {
        -1 => "",
        0 => "_fft",
        1 => "_tree",
        2 => "_poly",
        _ => panic!("Invalid enable option"),
    };

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let ramsim = RamConfig::new(&format!("{}{}", "sha256", kernel_name));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");
    const MSG_SIZE: usize = 8000;
    let mut msg = vec![0; MSG_SIZE as usize];
    for i in 0..MSG_SIZE - 1 {
        msg[i] = i as u8;
    }
    let msg_bits = array_to_bits(&msg);
    let len = msg.len() * 8;

    let mut hasher = Sha256::new();
    hasher.update(msg);
    let hash = hasher.finalize();

    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let mut builder = CircuitBuilder::<F, D>::new(CircuitConfig::standard_recursion_zk_config());
    let targets = make_circuits(&mut builder, len.try_into().unwrap());
    let mut pw = PartialWitness::new();

    for i in 0..len {
        pw.set_bool_target(targets.message[i], msg_bits[i]);
    }

    let expected_res = array_to_bits(hash.as_slice());
    for i in 0..expected_res.len() {
        if expected_res[i] {
            builder.assert_one(targets.digest[i].target);
        } else {
            builder.assert_zero(targets.digest[i].target);
        }
    }

    println!(
        "Constructing inner proof with {} gates",
        builder.num_gates()
    );
    let data = builder.build::<C>();

    let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
    prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);

    info!("Total number of memreq: {}", sys.ramsim.op_cnt);
    info!("Total number of operations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("Ramsim finished");
}
