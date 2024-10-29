use core::panic;

use env_logger::Env;
use log::info;

use plonky2::field::types::Field;
use plonky2::iop::generator::generate_partial_witness;
use plonky2::iop::witness::{PartialWitness, WitnessWrite};
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2::plonk::circuit_data::CircuitConfig;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};

use clap::{value_parser, Arg, Command};
use unizk::config::arch_config::ARCH_CONFIG;
use unizk::config::enable_config::ENABLE_CONFIG;
use unizk::config::RamConfig;
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;

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
        .arg(
            Arg::new("ram kb")
                .long("rk")
                .default_value("-1")
                .value_parser(value_parser!(i32)),
        )
        .get_matches();
    let ram_size: &usize = args.get_one::<usize>("ram").unwrap();
    let tiles: &usize = args.get_one::<usize>("tiles").unwrap();
    let enable: &i32 = args.get_one::<i32>("enable").unwrap();
    let ram_kb: &i32 = args.get_one::<i32>("ram kb").unwrap();

    unsafe {
        ARCH_CONFIG.rdbuf_sz_kb = ram_size * 1024 / 2;
        ARCH_CONFIG.wrbuf_sz_kb = ram_size * 1024 / 2;
        ARCH_CONFIG.num_tiles = *tiles;
        if *ram_kb >= 0 {
            ARCH_CONFIG.rdbuf_sz_kb = (ram_kb / 2) as usize;
            ARCH_CONFIG.wrbuf_sz_kb = (ram_kb / 2) as usize;
        }

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
    env_logger::Builder::from_env(Env::default().default_filter_or("debug")).init();
    let mut ramsim = RamConfig::new(&format!("{}{}", "fibonacci", kernel_name));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig::standard_recursion_config();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // The arithmetic circuit.
    let initial_a = builder.add_virtual_target();
    let initial_b = builder.add_virtual_target();
    let mut prev_target = initial_a;
    let mut cur_target = initial_b;
    for _ in 0..(1 << 20) - 1 {
        let temp = builder.add(prev_target, cur_target);
        prev_target = cur_target;
        cur_target = temp;
    }

    // Public inputs are the two initial values (provided below) and the result (which is generated).
    builder.register_public_input(initial_a);
    builder.register_public_input(initial_b);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial_a, F::ZERO);
    pw.set_target(initial_b, F::ONE);

    let mut data = builder.build::<C>();

    // data.prover_only.generators.clear();
    let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
    println!("common_data.fri_params: {:?}", data.common.fri_params);
    prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);

    info!("Total number of mem reqs: {}", sys.ramsim.op_cnt);
    info!("Total computations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}
