#!/bin/bash

simulators=(ramulatorv2)
patterns=(stream random)

for simulator in ${simulators[@]}; 
do
  for pattern in ${patterns[@]}; 
  do
    if [ ${simulator} == dramsim2 ]
    then
      python3 trace_generator.py --pattern ${pattern} --ratio 0.8 --num_reqs 5000000 --type ${simulator} --output ./mase_${pattern}_5M_R8W2_${simulator}.trace
    else
      python3 trace_generator.py --pattern ${pattern} --ratio 0.8 --num_reqs 5000000 --type ${simulator} --output ./${pattern}_5M_R8W2_${simulator}.trace
    fi
  done 
done