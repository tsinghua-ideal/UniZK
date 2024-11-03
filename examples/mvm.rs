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
use unizk::util::set_config;

const INPUT_SIZE: usize = 3000;
const OUTPUT_SIZE: usize = 3000;

fn main() {
    set_config();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let ramsim = RamConfig::new(&format!("mvm"));
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
