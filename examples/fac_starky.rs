use env_logger::Env;
use log::info;

use plonky2::field::types::Field;
use plonky2::plonk::config::{GenericConfig, PoseidonGoldilocksConfig};
use unizk::config::RamConfig;
use unizk::kernel::vector_operation::{VecOpConfig, VecOpSrc, VecOpType};
use unizk::memory::memory_allocator::MemAlloc;
use unizk::starky::constraint_consumer::ConstraintConsumer;
use unizk::starky::stark::EvaluationFrame;
use unizk::system::system::System;
use unizk::util::{BATCH_SIZE, SIZE_F};
use starky::config::StarkConfig;
use starky::factorial_stark::FactorialStark;

use unizk::starky::prover::prove;

const D: usize = 2;
type C = PoseidonGoldilocksConfig;
type F = <C as GenericConfig<D>>::F;
type S = FactorialStark<F, D>;

fn factorial<F: Field>(n: usize) -> F {
    (2..n + 1).fold(F::ONE, |x, i| x * F::from_canonical_u64(i as u64))
}

fn eval_packed_generic(
    sys: &mut System,
    vars: &EvaluationFrame,
    yield_constr: &ConstraintConsumer,
) -> Vec<VecOpConfig> {
    let mut res = Vec::new();
    let addr_constraint = sys.mem.get_addr("constraint").unwrap();
    let addr_local_values = vars.addr_local_values;
    let addr_next_values = vars.addr_next_values;
    let addr_public_inputs = vars.addr_public_inputs;

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_public_inputs + S::PI_INDEX_RES * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VS,
    });
    res.extend(yield_constr.constraint_last_row(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_values,
        addr_input_1: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_constraint,
        addr_input_1: 0, // one
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint_transition(addr_constraint));

    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_local_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_local_values,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::MUL,
        op_src: VecOpSrc::VV,
    });
    res.push(VecOpConfig {
        vector_length: BATCH_SIZE,
        addr_input_0: addr_next_values + BATCH_SIZE * SIZE_F,
        addr_input_1: addr_constraint,
        addr_output: addr_constraint,
        is_final_output: false,
        op_type: VecOpType::SUB,
        op_src: VecOpSrc::VV,
    });
    res.extend(yield_constr.constraint(addr_constraint));

    res
}

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info")).init();

    let mut ramsim = RamConfig::new(&format!("{}", "fac_starky"));
    // ramsim.txt_output = true;
    let mem = MemAlloc::new(256, 4096);
    let mut sys = System::new(mem, ramsim);

    let config = StarkConfig::standard_fast_config();
    let num_rows = 1 << 20;
    let public_inputs = [factorial(num_rows)];
    let stark = S::new(num_rows);
    let trace = stark.generate_trace();
    prove::<F, C, S, D>(
        &mut sys,
        stark,
        &config,
        trace,
        &public_inputs,
        eval_packed_generic,
    );

    info!("Total number of mem reqs: {}", sys.ramsim.op_cnt);
    info!("Total computations: {:?}", sys.get_computation());

    info!("Simulator finished");
    sys.ramsim.static_();

    info!("Start Ramsim");
    sys.ramsim.run();
    info!("End Ramsim");
}