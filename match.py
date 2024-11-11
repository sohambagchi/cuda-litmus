import pandas as pd
import argparse
import json

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", help="File to parse")

  args = parser.parse_args()
  df = pd.read_csv(args.file_path)

  test_summaries = {}
  test_summaries["all"] = {"tests": 0, "ptx_disallowed": 0, "ptx_mca_disallowed": 0, "ampere_seen": 0, "hopper_seen": 0}
  for _, row in df.iterrows():
    test = row["test"]
    test_base = test.split("_")[0]
    if test_base == "wrw":
      test_base = "wrw+2w"
    if test_base == "paper":
      test_base = "paper-example"
      
    if test_base not in test_summaries:
      test_summaries[test_base] = {"tests": 0, "ptx_disallowed": 0, "ptx_mca_disallowed": 0, "ampere_seen": 0, "hopper_seen": 0}

    test_summaries[test_base]["tests"] += 1
    test_summaries["all"]["tests"] += 1

    if row["PTX"] == "disallowed":
      test_summaries[test_base]["ptx_disallowed"] += 1
      test_summaries["all"]["ptx_disallowed"] += 1

    if row["PTX_MCA"] == "disallowed":
      test_summaries[test_base]["ptx_mca_disallowed"] += 1
      test_summaries["all"]["ptx_mca_disallowed"] += 1

    if row["ampere"] == "seen":
      if row["PTX_MCA"] == "disallowed":
        print(f"Unexpected weak test on Ampere: {test}")

      test_summaries[test_base]["ampere_seen"] += 1
      test_summaries["all"]["ampere_seen"] += 1

    if row["hopper"] == "seen":
      if row["PTX_MCA"] == "disallowed":
        print(f"Unexpected weak test on Hopper: {test}")

      test_summaries[test_base]["hopper_seen"] += 1
      test_summaries["all"]["hopper_seen"] += 1

  print(json.dumps(test_summaries, indent=2))

if __name__ == "__main__":
    main()

