import pandas as pd
import argparse

def analyze(file_path):
  # Initialize the dictionary to store test results
  test_summaries = {}

  unexpected_non_weak = set()
  unexpected_weak = set()
  same_tb_weak = set() # pull these out separately because we haven't seen it yet
  a_block_f_sys_dev_weak = set()

  test_summaries["all"] = {"tests": 0, "expected_weak": 0, "actual_weak": 0, "expected_non_weak": 0, "actual_non_weak": 0,  "weak_behaviors": 0, "total_behaviors": 0}

  # Reading the file
  with open(file_path, 'r') as file:
    lines = file.readlines()

  # Parsing the file content
  first_iteration = True
  total_iterations = 0
  for line in lines:
    # Identify and add tests to the dictionary with initial values for "weak" and "total"
    if line.strip().startswith("Test"):
      parts = line.strip().split()
      test_name = parts[1]  # Test name
      test_base = test_name.split("-")[0]

      if first_iteration:
        # Tests that are block scoped or relaxed can see valid weak behaviors
        # We set it to all of these first and remove them as we see weak behaviors
        if test_base not in test_summaries:
          test_summaries[test_base] = {"tests": 0, "expected_weak": 0, "actual_weak": 0, "expected_non_weak": 0, "actual_non_weak": 0,  "weak_behaviors": 0, "total_behaviors": 0}

        if "SCOPE_BLOCK" in test_name  or "RELAXED" in test_name or "STORE_SC" in test_name:
          test_summaries[test_base]["expected_weak"] += 1
          test_summaries["all"]["expected_weak"] += 1

          unexpected_non_weak.add(test_name)
        else:
          test_summaries[test_base]["expected_non_weak"] += 1
          test_summaries[test_base]["actual_non_weak"] += 1
          test_summaries["all"]["expected_non_weak"] += 1
          test_summaries["all"]["actual_non_weak"] += 1



        test_summaries[test_base]["tests"] += 1
        test_summaries["all"]["tests"] += 1

      weak_value = int(parts[3].strip(','))  # Extract the "weak" value
      total_value = int(parts[5].strip(','))  # Extract the "total" value

      if weak_value > 0:
        if test_name in unexpected_non_weak:
         test_summaries[test_base]["actual_weak"] += 1
         test_summaries["all"]["actual_weak"] += 1

         unexpected_non_weak.remove(test_name)
        if ("TB_012" in test_name or "TB_0123" in test_name) and "TB_012_3" not in test_name:
          same_tb_weak.add(test_name)
        if "SCOPE_BLOCK-FENCE_SCOPE_DEVICE" in test_name or "SCOPE_BLOCK-FENCE_SCOPE_SYSTEM" in test_name:
          a_block_f_sys_dev_weak.add(test_name)
        # Tests that are not block scoped or relaxed should not see weak behaviors
        if "SCOPE_BLOCK" not in test_name and "RELAXED" not in test_name and "STORE_SC" not in test_name:
          test_summaries[test_base]["actual_non_weak"] -= 1
          test_summaries["all"]["actual_non_weak"] -= 1
          unexpected_weak.add(test_name)
      test_summaries[test_base]["weak_behaviors"] += weak_value
      test_summaries[test_base]["total_behaviors"] += total_value
      test_summaries["all"]["weak_behaviors"] += weak_value
      test_summaries["all"]["total_behaviors"] += total_value
    elif line.strip().startswith("Iteration"):
        total_iterations += 1
        parts = line.strip().split()
        if parts[1] == "1":
          first_iteration = False

  # Convert dictionary to a pandas DataFrame for better visualization
  df_summaries = pd.DataFrame.from_dict(test_summaries, orient='index')

  print(f"Total iterations: {total_iterations}")

  print(df_summaries)
  #print(df_results)

  # only print out non same tb unexpected non weak tests
  for test_name in list(unexpected_non_weak): 
    if "TB_012" in test_name or "TB_0123" in test_name:
      unexpected_non_weak.remove(test_name)
#  print(f"Unexpected non-weak tests: {unexpected_non_weak}")

#  print(f"Unexpected weak tests: {unexpected_weak}")

  print(f"Weak same threadblock tests: {same_tb_weak}")
#  print(f"Weak atomic sys/device, fence block: {a_block_f_sys_dev_weak}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="File to parse")
    args = parser.parse_args()
    analyze(args.file_path)

if __name__ == "__main__":
    main()
