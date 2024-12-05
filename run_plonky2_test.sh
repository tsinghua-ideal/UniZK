cp configs/Cargo.toml Cargo.toml
cargo run -r --example factorial
rm traces/factorial.bin traces/factorial.txt
cargo run -r --example fibonacci
rm traces/fibonacci.bin traces/fibonacci.txt
cargo run -r --example mvm
rm traces/mvm.bin traces/mvm.txt
cargo clean
cp configs/Cargo.toml.v0.1 Cargo.toml
cargo run -r --example sha256
rm traces/sha256.bin traces/sha256.txt
cargo run -r --example ecdsa
rm traces/ecdsa.bin traces/ecdsa.txt
cargo run -r --example img_crop
rm traces/crop.bin traces/crop.txt
cargo clean