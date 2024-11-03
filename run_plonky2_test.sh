cp configs/Cargo.toml Cargo.toml
cargo run -r --example factorial
cargo run -r --example fibonacci
cargo run -r --example mvm
cargo clean
cp configs/Cargo.toml.v0.1 Cargo.toml
cargo run -r --example sha256
cargo run -r --example ecdsa
cargo run -r --example img_crop
cargo clean