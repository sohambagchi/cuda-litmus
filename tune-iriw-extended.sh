#!/bin/bash

## Tune specific versions of tests

PARAM_FILE="params.txt"
RESULT_DIR="results"
SHADER_DIR="kernels"
PARAMS_DIR="params"
TARGET_DIR="target"

function make_even() {
    if (( $1 % 2 == 0 )); then
        echo "$1"
    else
	echo "$(($1 + 1))"
    fi
}

# Generate a random number between min and max
function random_between() {
  local min=$1
  local max=$2

  local range=$((max - min + 1))
  local random=$((RANDOM % range + min))
  echo "$random"
}

function random_config() {
  local workgroupLimiter=$1
  local workgroupSizeLimiter=$2

  echo "testIterations=1000" > $PARAM_FILE
  local testingWorkgroups=$(random_between 4 $workgroupLimiter)
  echo "testingWorkgroups=$testingWorkgroups" >> $PARAM_FILE
  local maxWorkgroups=$(random_between $testingWorkgroups $workgroupLimiter)
  echo "maxWorkgroups=$maxWorkgroups" >> $PARAM_FILE
  # ensures total threads is divisible by 2
  local workgroupSize=$(make_even $(random_between 1 $workgroupSizeLimiter))
  echo "workgroupSize=$workgroupSize" >> $PARAM_FILE
  echo "shufflePct=$(random_between 0 100)" >> $PARAM_FILE
  echo "barrierPct=$(random_between 0 100)" >> $PARAM_FILE
  local stressLineSize=$(( $(random_between 2 10) ** 2 ))
  echo "stressLineSize=$stressLineSize" >> $PARAM_FILE
  local stressTargetLines=$(random_between 1 16)
  echo "stressTargetLines=$stressTargetLines" >> $PARAM_FILE
  echo "scratchMemorySize=$((32 * $stressLineSize * $stressTargetLines))" >> $PARAM_FILE
  echo "memStride=$(random_between 1 7)" >> $PARAM_FILE
  echo "memStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "memStressIterations=$(random_between 0 1024)" >> $PARAM_FILE
  echo "memStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "preStressPct=$(random_between 0 100)" >> $PARAM_FILE
  echo "preStressIterations=$(random_between 0 128)" >> $PARAM_FILE
  echo "preStressPattern=$(random_between 0 3)" >> $PARAM_FILE
  echo "stressAssignmentStrategy=$(random_between 0 1)" >> $PARAM_FILE
  echo "permuteThread=419" >> $PARAM_FILE
}

function run_test() {
  local test=$1
  local tb=$2
  local scope=$3
  local fence_scope=$4
  local variant=$5
  local params=$6
  res=$(./$TARGET_DIR/$test-$tb-$scope-$fence_scope-$variant-runner -s $PARAM_FILE -t $PARAMS_DIR/$params)
  local x_y_weak=$(echo "$res" | tail -n 1 | sed 's/.*x y weak: \(.*\)$/\1/')
  local a_b_weak=$(echo "$res" | tail -n 2 | head -n 1 | sed 's/.*a b weak: \(.*\)$/\1/')
  local weak_behaviors=$(echo "$res" | tail -n 3 | head -n 1 | sed 's/.*of weak behaviors: \(.*\)$/\1/')
  local total_behaviors=$(echo "$res" | tail -n 4 | head -n 1 | sed 's/.*Total behaviors: \(.*\)$/\1/')
  local weak_rate=$(echo "$res" | tail -n 5 | head -n 1 | sed 's/.*rate: \(.*\) per second/\1/')

  echo "  Test $test-$tb-$scope-$fence_scope-$variant weak: $weak_behaviors, x_y_weak: $x_y_weak, a_b_weak: $a_b_weak, total: $total_behaviors, rate: $weak_rate per second"

  if awk "BEGIN {exit !($weak_rate > 0)}"; then
    local test_result_dir="$RESULT_DIR/$test-$tb-$scope-$fence_scope-$variant"
    if [ ! -d "$test_result_dir" ] ; then
      mkdir "$test_result_dir"
      cp $PARAM_FILE "$test_result_dir"
      echo $weak_rate > "$test_result_dir/rate"
    else
      local max_rate=$(cat "$test_result_dir/rate")
      if awk "BEGIN {exit !($weak_rate > $max_rate)}"; then
        cp $PARAM_FILE "$test_result_dir"
        echo $weak_rate > "$test_result_dir/rate"
      fi
    fi
  fi

}

if [ $# -lt 1 ] ; then
  echo "Need to pass file with test combinations"
  exit 1
fi

if [ ! -d "$RESULT_DIR" ] ; then
  mkdir $RESULT_DIR
fi

if [ ! -d "$TARGET_DIR" ] ; then
  mkdir $TARGET_DIR
fi

tuning_file=$1

compile=true
if [ $# == 2 ] ; then
  compile=false
fi

readarray tests < $tuning_file

if "$compile"; then
for test in "${tests[@]}"; do
  set -- $test
  test_name=$1
  tb=$3
  scope=$4
  f_scope=$5
  variant=$6

  echo "Compiling $test_name-$tb-$scope-$f_scope-$variant runner"
  nvcc -D$tb -D$scope -D$f_scope -D$variant -I. -rdc=true -arch sm_80 iriw-extended-runner.cu "kernels/$test_name.cu" -o "$TARGET_DIR/$test_name-$tb-$scope-$f_scope-$variant-runner"
done
fi

iter=0

while [ true ]
do
  echo "Iteration: $iter"
  random_config 1024 256
  for test in "${tests[@]}"; do
    set -- $test
    test_name=$1
    test_params=$2
    tb=$3
    scope=$4
    f_scope=$5
    variant=$6

    run_test $test_name $tb $scope $f_scope $variant $test_params
  done
  iter=$((iter + 1))
done

