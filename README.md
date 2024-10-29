This is the simulator for UniZK, an accelerator for  Zero-Knowledge Proof with unified hardware and flexible kernel mapping.

## Thirdparty
The codes of the following third-party libraries are sourced from:
- imgcrop: <https://github.com/ChickenLover/plonky2-zkedit>
- plonky2: <https://github.com/0xPolygonZero/plonky2>
- proto-neural-zkp: <https://github.com/worldcoin/proto-neural-zkp>
- ramulator2: <https://github.com/CMU-SAFARI/ramulator2>
- sha256: <https://github.com/polymerdao/plonky2-sha256>
- sha256-starky: <https://github.com/tumberger/plonky2/tree/sha256-starky>

We appreciate the contributions of these open-source authors.

## How to get started
### Prerequiste
- rust 1.80
- g++-12
- cmake 3.22

### Build Ramulator2
```
cd thirdparty/ramulator2
mkdir build
cd build
cmake ..
make -j
```
### Run tests for plonky2
```
./run_plonky2_test.sh
./run_plonky2_test_2.sh
```
### Run tests for starky and recursive proof
```
./run_starky_test.sh
```

A log file with the name of the application, such as “sha256.log”, will appear in the folder with the simulation results.