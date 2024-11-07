#!/bin/bash

PARAMS_DIR=results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt

## Compile them

nvcc -DTB_01_23 -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DALL_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-all-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSTORES_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-stores-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DFIRST_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-first-load-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSECOND_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-second-load-sc-runner

nvcc -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/mp.cu -o target/mp-runner

## Run them

echo "iriw:"
./target/iriw-runner -s $PARAMS_DIR -t params/2-loc.txt
echo "iriw-sc:"
./target/iriw-all-sc-runner -s $PARAMS_DIR -t params/2-loc.txt
echo "iriw-stores-sc:"
./target/iriw-stores-sc-runner -s $PARAMS_DIR -t params/2-loc.txt
echo "iriw-first-load-sc:"
./target/iriw-first-load-sc-runner -s $PARAMS_DIR -t params/2-loc.txt
echo "iriw-second-load-sc:"
./target/iriw-second-load-sc-runner -s $PARAMS_DIR -t params/2-loc.txt
echo "mp:"
./target/mp-runner -s $PARAMS_DIR -t params/2-loc.txt
