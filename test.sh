#!/bin/bash

## Compile them

nvcc -DTB_01_23 -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DALL_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-all-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSTORES_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-stores-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DFIRST_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-first-load-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSECOND_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-seonc-load-sc-runner

nvcc -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/mp.cu -o target/mp-runner

## Run them

./target/iriw-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt
./target/iriw-all-sc-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt
./target/iriw-stores-sc-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt
./target/iriw-first-load-sc-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt
./target/iriw-second-load-sc-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt
./target/mp-runner -s results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt