set -e
mkdir -p ../build
cd ../build
cmake ..
make -j
cd -
echo "------------------------------" | tee stdout
time ./ramsim -f trace.yaml | tee -a stdout
