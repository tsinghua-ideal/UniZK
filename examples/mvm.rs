use env_logger::Env;
use log::info;

use plonky2::iop::generator::generate_partial_witness;

use clap::{value_parser, Arg, Command};
use rand::Rng;
use unizk::config::enable_config::ENABLE_CONFIG;
use unizk::config::{arch_config::ARCH_CONFIG, RamConfig};
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;

use neural_zkp::{Circuit, Options};

const INPUT_SIZE: usize = 3000;
const OUTPUT_SIZE: usize = 3000;

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

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let ramsim = RamConfig::new(&format!("mvm_{}_{}{}", ram_size, tiles, kernel_name));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");

    let options = Options {
        bench: false,
        input_size: INPUT_SIZE,
        output_size: OUTPUT_SIZE,
        coefficient_bits: 16,
        num_wires: 400,
        num_routed_wires: 400,
        constant_gate_size: 90,
    };
    let mut rng = rand::thread_rng();
    let quantize_coeff = |c: i32| c % (1 << options.coefficient_bits);
    let coefficients: Vec<i32> = (0..options.input_size * options.output_size)
        .map(|_| quantize_coeff(rng.gen()))
        .collect();

    let circuit = Circuit::build(&options, &coefficients);
    let input_values = (0..options.input_size as i32)
        .into_iter()
        .map(|_| rng.gen())
        .collect::<Vec<_>>();
    let pw = circuit.get_pw(&input_values);

    println!("circuit config: {:?}", &circuit.data.common.config);
    // data.prover_only.generators.clear();
    let partition_witness =
        generate_partial_witness(pw, &circuit.data.prover_only, &circuit.data.common);
    prove_with_partition_witness(
        &mut sys,
        &circuit.data.prover_only,
        &circuit.data.common,
        partition_witness,
    );

    info!("Total number of operations: {}", sys.ramsim.op_cnt);

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("sys.computation: {:?}", sys.get_computation());

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("Ramsim finished");
}
