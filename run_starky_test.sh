cp configs/Cargo.toml Cargo.toml
cargo run -r --example aes_starky
cargo run -r --example aes_starky_recursive
cargo run -r --example sha256_starky -- 126
cargo run -r --example sha256_starky_recursive -- 126
cargo run -r --example fib_starky
cargo run -r --example fib_starky_recursive
cargo run -r --example fac_starky
cargo run -r --example fac_starky_recursive
cargo clean