use anyhow::{Context, Ok, Result};
use log::{info, Level, LevelFilter};
use maybe_rayon::rayon;
use plonky2::{
    field::{secp256k1_scalar::Secp256K1Scalar, types::Sample},
    iop::witness::PartialWitness,
    plonk::{
        circuit_builder::CircuitBuilder,
        circuit_data::{CircuitConfig, CircuitData},
        config::{GenericConfig, PoseidonGoldilocksConfig},
    },
};
use plonky2_ecdsa::{
    curve::{
        curve_types::{Curve, CurveScalar},
        ecdsa::{sign_message, ECDSAPublicKey, ECDSASecretKey, ECDSASignature},
        secp256k1::Secp256K1,
    },
    gadgets::{
        curve::CircuitBuilderCurve,
        ecdsa::{verify_message_circuit, ECDSAPublicKeyTarget, ECDSASignatureTarget},
        nonnative::CircuitBuilderNonNative,
    },
};
use plonky2_field::goldilocks_field::GoldilocksField;

pub fn get_circuit() -> (
    CircuitData<GoldilocksField, PoseidonGoldilocksConfig, 2>,
    PartialWitness<GoldilocksField>,
) {
    const D: usize = 2;
    type C = PoseidonGoldilocksConfig;
    type F = <C as GenericConfig<D>>::F;
    type Curve = Secp256K1;

    let config = CircuitConfig::standard_ecc_zk_config();
    let pw = PartialWitness::new();
    let mut builder = CircuitBuilder::<F, D>::new(config);

    let msg = Secp256K1Scalar::rand();
    let msg_target = builder.constant_nonnative(msg);

    let sk = ECDSASecretKey::<Curve>(Secp256K1Scalar::rand());
    let pk = ECDSAPublicKey((CurveScalar(sk.0) * Curve::GENERATOR_PROJECTIVE).to_affine());

    let pk_target = ECDSAPublicKeyTarget(builder.constant_affine_point(pk.0));

    let sig = sign_message(msg, sk);

    let ECDSASignature { r, s } = sig;
    let r_target = builder.constant_nonnative(r);
    let s_target = builder.constant_nonnative(s);
    let sig_target = ECDSASignatureTarget {
        r: r_target,
        s: s_target,
    };
    verify_message_circuit(&mut builder, msg_target, sig_target, pk_target);
    println!("{sig:?}");
    let data = builder.build::<C>();
    (data, pw)
}

fn main() -> Result<()> {
    // Initialize logging
    let mut builder = env_logger::Builder::from_default_env();
    builder.format_timestamp(None);
    builder.filter_level(LevelFilter::Info);
    builder.try_init()?;

    // Run the benchmark
    let num_cpus = 1; //num_cpus::get();
    let threads = num_cpus;
    println!("Number of CPUs: {}", num_cpus);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Failed to build thread pool.")?
        .install(|| {
            info!(
                "Using {} compute threads on {} cores",
                rayon::current_num_threads(),
                num_cpus
            );
            // Run the benchmark. `options.lookup_type` determines which benchmark to run.
            let (data, pw) = get_circuit();
            let proof = data.prove(pw).unwrap();
            data.verify(proof).unwrap();
        });
    Ok(())
}
