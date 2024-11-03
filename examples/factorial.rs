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
use unizk::util::set_config;

fn main() {
    set_config();
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();
    let mut ramsim = RamConfig::new(&format!("{}", "factorial"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    info!("Starting simulator");
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;

    let config = CircuitConfig {
        // num_wires: 400,
        // num_routed_wires: 400,
        ..CircuitConfig::standard_recursion_config()
    };
    let mut builder = CircuitBuilder::<F, D>::new(config);

    // The arithmetic circuit.
    let initial = builder.add_virtual_target();
    let mut cur_target = initial;
    for i in 2..(1 << 20) {
        let i_target = builder.constant(F::from_canonical_u32(i));
        cur_target = builder.mul(cur_target, i_target);
    }

    // Public inputs are the initial value (provided below) and the result (which is generated).
    builder.register_public_input(initial);
    builder.register_public_input(cur_target);

    // Provide initial values.
    let mut pw = PartialWitness::new();
    pw.set_target(initial, F::ONE);

    let mut data = builder.build::<C>();

    // data.prover_only.generators.clear();
    let partition_witness = generate_partial_witness(pw, &data.prover_only, &data.common);
    println!("common_data.fri_params: {:?}", data.common.fri_params);
    prove_with_partition_witness(&mut sys, &data.prover_only, &data.common, partition_witness);

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}
