use anyhow::{Context, Ok, Result};
use log::{info, Level, LevelFilter};
use plonky2::iop::generator::generate_partial_witness;
use rand::Rng;

use neural_zkp::{Circuit, Options};

const INPUT_SIZE: usize = 3000;
const OUTPUT_SIZE: usize = 3000;

fn run() {
    let options = Options {
        bench:              false,
        input_size:         INPUT_SIZE,
        output_size:        OUTPUT_SIZE,
        coefficient_bits:   16,
        num_wires:          400,
        num_routed_wires:   400,
        constant_gate_size: 90,
    };
    let mut rng = rand::thread_rng();
    let quantize_coeff = |c: i32| c % (1 << options.coefficient_bits);
    let coefficients: Vec<i32> = (0..options.input_size * options.output_size)
        .map(|_| quantize_coeff(rng.gen()))
        .collect();

    let circuit = Circuit::build(&options, &coefficients);

    // Set witness for proof
    let input_values = (0..options.input_size as i32)
        .into_iter()
        .map(|_| rng.gen())
        .collect::<Vec<_>>();
    let proof = circuit.prove(&input_values).unwrap();
}

fn main() {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Info);
    builder.try_init().unwrap();

    let num_cpus = num_cpus::get();
    let threads = num_cpus;
    println!("Number of CPUs: {}", num_cpus);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build thread pool.")
        .unwrap()
        .install(|| {
            info!(
                "Using {} compute threads on {} cores",
                rayon::current_num_threads(),
                num_cpus
            );
            // Run the benchmark. `options.lookup_type` determines which benchmark to run.
            run()
        });
}
