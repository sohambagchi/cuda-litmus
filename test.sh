#!/bin/bash

PARAMS_DIR=results/iriw-TB_01_23-SCOPE_DEVICE-NO_FENCE-ACQUIRE/params.txt

## Compile them

nvcc -DTB_01_23 -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DALL_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-all-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSTORES_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-stores-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DFIRST_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-first-load-sc-runner

nvcc -DTB_01_23 -DSCOPE_DEVICE -DSECOND_LOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/iriw-sc.cu -o target/iriw-second-load-sc-runner

nvcc -DSCOPE_DEVICE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/mp.cu -o target/mp-runner

nvcc -DTB_0_1_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner

nvcc -DTB_0_1_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner


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

nvcc -DTB_0_1_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner

nvcc -DTB_0_1_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner

echo "rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE:"
./target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner -s results/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt

echo "rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE:"
./target/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner -s results/rwc-TB_0_1_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt


nvcc -DTB_01_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner

nvcc -DTB_01_2 -DSCOPE_SYSTEM -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner

echo "rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE:"
./target/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner -s results/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt

echo "rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE:"
./target/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner -s results/rwc-TB_01_2-SCOPE_SYSTEM-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt


nvcc -DTB_01_2 -DSCOPE_DEVICE -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner

nvcc -DTB_01_2 -DSCOPE_DEVICE -DFENCE_SCOPE_DEVICE -DTHREAD_2_FENCE -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner

echo "rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE:"
./target/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-runner -s results/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt

echo "rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE:"
./target/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE-THREAD_0_STORE_RELEASE-runner -s results/rwc-TB_01_2-SCOPE_DEVICE-FENCE_SCOPE_DEVICE-THREAD_2_FENCE/params.txt -t params/2-loc.txt


nvcc -DTB_0_12 -DSCOPE_DEVICE -DLOAD_SC -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC-runner

nvcc -DTB_0_12 -DSCOPE_DEVICE -DLOAD_SC -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/rwc.cu -o target/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC-THREAD_0_STORE_RELEASE-runner

echo "rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC:"
./target/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC-runner -s results/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC/params.txt -t params/2-loc.txt

echo "rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC-THREAD_0_STORE_RELEASE:"
./target/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC-THREAD_0_STORE_RELEASE-runner -s results/rwc-TB_0_12-SCOPE_DEVICE-NO_FENCE-LOAD_SC/params.txt -t params/2-loc.txt


nvcc -DTB_01_2 -DSCOPE_SYSTEM -DACQUIRE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/wrc.cu -o target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE-runner

nvcc -DTB_01_2 -DSCOPE_SYSTEM -DACQUIRE -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/wrc.cu -o target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE-THREAD_0_STORE_RELEASE-runner

echo "wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE:"
./target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE-runner -s results/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt

echo "wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE-THREAD_0_STORE_RELEASE:"
./target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE-THREAD_0_STORE_RELEASE-runner -s results/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt


nvcc -DTB_01_2 -DSCOPE_SYSTEM -DACQ_REL -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/wrc.cu -o target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-runner

nvcc -DTB_01_2 -DSCOPE_SYSTEM -DACQ_REL -DTHREAD_0_STORE_RELEASE -I. -rdc=true -arch sm_80 runner.cu functions.cu kernels/wrc.cu -o target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-THREAD_0_STORE_RELEASE-runner

echo "wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL:"
./target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-runner -s results/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt

echo "wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-THREAD_0_STORE_RELEASE:"
./target/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQ_REL-THREAD_0_STORE_RELEASE-runner -s results/wrc-TB_01_2-SCOPE_SYSTEM-NO_FENCE-ACQUIRE/params.txt -t params/2-loc.txt