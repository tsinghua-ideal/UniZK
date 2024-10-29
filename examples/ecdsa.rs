mod ecdsa_veri;
use env_logger::Env;
use log::info;

use ecdsa_veri::get_circuit;
use plonky2::iop::generator::generate_partial_witness;
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
    let ramsim = RamConfig::new(&format!("{}{}", "ecdsa", kernel_name));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");
    let (data, pw) = get_circuit();

    let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
    prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);

    info!("Total number of memreq: {}", sys.ramsim.op_cnt);
    println!("Total number of operations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("Ramsim finished");
}
