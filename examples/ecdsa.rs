mod cpu_ecdsa;
use env_logger::Env;
use log::info;

use cpu_ecdsa::get_circuit;
use plonky2::iop::generator::generate_partial_witness;
use unizk::config::RamConfig;
use unizk::memory::memory_allocator::MemAlloc;
use unizk::plonk::prover::prove_with_partition_witness;
use unizk::system::system::System;
use unizk::{config::enable_config::ENABLE_CONFIG, util::set_config};

use clap::{value_parser, Arg, Command};

fn main() {
    set_config();

    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let ramsim = RamConfig::new(&format!("{}", "ecdsa"));
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
