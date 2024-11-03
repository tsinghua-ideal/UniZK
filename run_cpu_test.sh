cp configs/Cargo.toml Cargo.toml
cargo run -r --example cpu_factorial > cpu_factorial.log
cargo run -r --example cpu_fibonacci > cpu_fibonacci.log
cargo run -r --example cpu_mvm > cpu_mvm.log
cargo run -r --example cpu_aes_starky > cpu_aes_starky.log
cargo run -r --example cpu_sha256_starky -- 126 > cpu_sha256_starky_126_block.log
cargo run -r --example cpu_sha256_starky -- 1 > cpu_sha256_starky_1_block.log
cargo run -r --example cpu_fib_starky > cpu_fibonacci_starky.log
cargo run -r --example cpu_fac_starky > cpu_factorial_starky.log
cargo clean
cp configs/Cargo.toml.v0.1 Cargo.toml
cargo run -r --example cpu_sha256 > cpu_sha256.log
cargo run -r --example cpu_ecdsa > cpu_ecdsa.log
cargo run -r --example cpu_img_crop > cpu_img_crop.log
cargo clean