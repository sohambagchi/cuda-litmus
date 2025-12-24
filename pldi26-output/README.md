This directory contains the raw output of the litmus testing for the paper. In particular, the files are:

## a100.txt

This file contains the output of running all tests on an NVIDIA A100 (Ampere) GPU, for 67 iterations.

# gh200.txt

This file contains the output of running all tests on an NVIDIA GH200 (Grace Hopper) GPU, for 4 iterations.

# h100.txt

This file contains the output of running all tests on an NVIDIA H100 (Hopper) GPU, for 95 iterations.
TODO: I need to run the buggy variants from last time, as well as the new Z6-3/2+2W/3.2W variants, on this machine.
I also need to run the paper examples and the counterexample.


In addition, we ran the examples in the paper (Figure 2 and its variant, the counterexample submitted to NVIDIA) on all 3 GPUs, and saw no violations.

