make -C ../build -j
g++ accel.cpp -I ../../ -I ../../ramsim/src -I ../../ramsim/ext/spdlog/include -I ../../ramsim/ext/yaml-cpp/include --std=c++20 -L../ -lramulator -Wl,-rpath ../ -o accel
./accel | tee stdout
