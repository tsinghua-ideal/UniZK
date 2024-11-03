This is the simulator for UniZK, an accelerator for  Zero-Knowledge Proof with unified hardware and flexible kernel mapping.

## Thirdparty
The codes of the following third-party libraries are sourced from:
- ecdsa: <https://github.com/0xPolygonZero/plonky2-ecdsa/tree/bdb6504bca250db1548cdcce2407d4e334990c33>
- imgcrop: <https://github.com/ChickenLover/plonky2-zkedit/tree/d20567361bd405716f6736a88dc263978f267d62>
- plonky2: <https://github.com/0xPolygonZero/plonky2/tree/30b47998262642be54da5acf03dfca31af4d93f7>
- plonky2v0.1: <https://github.com/polymerdao/plonky2/tree/4cb0b48df1d227d5461a4c28ed025aaea64e2e62>
- proto-neural-zkp: <https://github.com/worldcoin/proto-neural-zkp/tree/b2b514ac0857fd5e1cb5da9399fcd6020b1730e3>
- ramsim: RamSim is an enhanced version of Ramulator2, adding support for simulating computation latency and data dependencies. Most of its source code is derived from <https://github.com/CMU-SAFARI/ramulator2/tree/b7c70275f04126c647edb989270cc429776955d1>.
- sha256: <https://github.com/polymerdao/plonky2-sha256/tree/06d128e78ed8d29b21d58294b069e852c1866f8d>
- sha256-starky: <https://github.com/tumberger/plonky2/tree/474ed82c385b446f89bc18e648b5ad7d5d94ce06>

We appreciate the contributions of these open-source authors.

## How to get started
### Prerequiste
- rust 1.80
- g++ 11.4
- cmake 3.22

### Use a nightly toolchain for Plonky2
```
rustup override set nightly
```

### Build RamSim
```
cd thirdparty/ramsim
mkdir build
cd build
cmake ..
make -j
```
### Run tests on CPU
```
./run_cpu_test.sh
```

### Run tests for plonky2
```
./run_plonky2_test.sh
```
### Run tests for starky and recursive proof
```
./run_starky_test.sh
```

A log file with the name of the application, such as “sha256.log”, will appear in the folder with the simulation results.

## Notes
### Disk space
Trace files can take up a lot of disk space. Ensure you have enough space when testing large applications and remember to clean up the files after testing!

### GPU
The GPU performance can be evaluated using the GPU implementation of Plonky2 (available at <https://github.com/sideprotocol/plonky2-gpu>) with our CPU-based application code located in the `examples` directory.