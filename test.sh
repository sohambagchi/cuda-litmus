#!/bin/bash

nvcc -DTB_01_23 -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-sc-runner

nvcc -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/mp.cu -o target/mp-runner

