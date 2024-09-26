import pandas as pd
import argparse

def analyze(file_path):
  # Initialize the dictionary to store test results
  test_results = {}

  total_expected_weak = 0
  unexpected_non_weak = set()
  total_expected_non_weak = 0
  unexpected_weak = set()
  same_tb_weak = set() # pull these out separately because we haven't seen it yet
  a_block_f_sys_dev_weak = set()

  test_results["all"] = {"weak": 0, "total": 0}

  # Reading the file
  with open(file_path, 'r') as file:
    lines = file.readlines()

  # Parsing the file content
  for line in lines:
    # Identify and add tests to the dictionary with initial values for "weak" and "total"
    if line.startswith("Compiling"):
      test_name = line.split()[1]  # Extract the test name
      # Tests that are block scoped or relaxed can see valid weak behaviors
      # We set it to all of these first and remove them as we see weak behaviors
      if "SCOPE_BLOCK" in test_name  or "RELAXED" in test_name or "STORE_SC" in test_name:
        total_expected_weak += 1
        unexpected_non_weak.add(test_name)
      else:
        total_expected_non_weak += 1
      if test_name not in test_results:
        test_results[test_name] = {"weak": 0, "total": 0}
    # Identify the iteration results and update the dictionary
    elif line.strip().startswith("Test"):
      parts = line.strip().split()
      test_name = parts[1]  # Test name
      weak_value = int(parts[3].strip(','))  # Extract the "weak" value
      total_value = int(parts[5].strip(','))  # Extract the "total" value

      if weak_value > 0:
        if test_name in unexpected_non_weak:
          unexpected_non_weak.remove(test_name)
        if "TB_012" in test_name or "TB_0123" in test_name:
          same_tb_weak.add(test_name)
        if "SCOPE_BLOCK-FENCE_SCOPE_DEVICE" in test_name or "SCOPE_BLOCK-FENCE_SCOPE_SYSTEM" in test_name:
          a_block_f_sys_dev_weak.add(test_name)
        # Tests that are not block scoped or relaxed should not see weak behaviors
        if "SCOPE_BLOCK" not in test_name and "RELAXED" not in test_name and "STORE_SC" not in test_name:
          unexpected_weak.add(test_name)
      # Update the dictionary values
      if test_name in test_results:
        test_results[test_name]["weak"] += weak_value
        test_results[test_name]["total"] += total_value
        test_results["all"]["weak"] += weak_value
        test_results["all"]["total"] += total_value


  # Convert dictionary to a pandas DataFrame for better visualization
  df_results = pd.DataFrame.from_dict(test_results, orient='index')
  print(f"Total tests: {len(test_results) - 1}")
  print(df_results)

  print(f"Total expected weak tests: {total_expected_weak}")
  # only print out non same tb unexpected non weak tests
  for test_name in list(unexpected_non_weak): 
    if "TB_012" in test_name or "TB_0123" in test_name:
      unexpected_non_weak.remove(test_name)
  print(f"Unexpected non-weak tests: {unexpected_non_weak}")

  print(f"Total expected non-weak tests: {total_expected_non_weak}")
  print(f"Unexpected weak tests: {unexpected_weak}")

  print(f"Weak same threadblock tests: {same_tb_weak}")
  print(f"Weak atomic sys/device, fence block: {a_block_f_sys_dev_weak}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="File to parse")
    args = parser.parse_args()
    analyze(args.file_path)

if __name__ == "__main__":
    main()
