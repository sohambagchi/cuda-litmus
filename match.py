import pandas as pd
import argparse
import json

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", help="File to parse")

  args = parser.parse_args()
  df = pd.read_csv(args.file_path)

  machines = ["a100", "h100", "gh200"]

  test_summaries = {}
  test_summaries["all"] = {"tests": 0, "relaxed_tests": 0, "base_allowed": 0, "strengthened_allowed": 0, "base_matched": 0, "strengthened_matched": 0, "base_allowed_matched": 0, "strengthened_allowed_matched": 0, "base_disallowed_matched": 0, "strengthened_disallowed_matched": 0}

  for machine in machines:
      test_summaries["all"][machine] = {"base_allowed_seen": 0, "base_disallowed_seen": 0, "strengthened_allowed_seen": 0, "strengthened_disallowed_seen": 0, "relaxed_seen": 0, "base_matched": 0, "strengthened_matched": 0}

  for _, row in df.iterrows():
    test = row["test"]
    test_base = test.split("_")[0]
    if test_base == "wrw":
      test_base = "wrw+2w"
    if test_base == "paper":
      test_base = "paper-example"
    if test_base == "two":
      test_base = "2+2w"
    if test_base == "three":
      test_base = "3.2w"
    if test_base == "z6":
      test_base = "z6-3"

    if test_base not in test_summaries:
      test_summaries[test_base] = {"tests": 0, "relaxed_tests": 0, "base_allowed": 0, "strengthened_allowed": 0, "ampere_seen": 0, "hopper_seen": 0}
      for machine in machines:
        test_summaries[test_base][machine] = {"base_allowed_seen": 0, "base_disallowed_seen": 0, "strengthened_allowed_seen": 0, "strengthened_disallowed_seen": 0, "relaxed_seen": 0}

    if "RELAXED" in test:
      test_summaries[test_base]["relaxed_tests"] += 1
      test_summaries["all"]["relaxed_tests"] += 1
    test_summaries[test_base]["tests"] += 1
    test_summaries["all"]["tests"] += 1

    if row["base"] == "INSTANCE":
      test_summaries[test_base]["base_allowed"] += 1
      test_summaries["all"]["base_allowed"] += 1

    if row["strengthened"] == "INSTANCE":
      test_summaries[test_base]["strengthened_allowed"] += 1
      test_summaries["all"]["strengthened_allowed"] += 1

    base_matched = True
    base_allowed_matched = True
    base_disallowed_matched = True
    strengthened_matched = True
    strengthened_allowed_matched = True
    strengthened_disallowed_matched = True
    for machine in machines:
      if row[machine] == "seen":
        if "RELAXED" in test:
          test_summaries["all"][machine]["relaxed_seen"] += 1
          test_summaries[test_base][machine]["relaxed_seen"] += 1
        if row["strengthened"] == "UNSAT":
          strengthened_matched = False
          strengthened_disallowed_matched = False
          print(f"Unexpected weak test on {machine} under strengthened: {test}")
          test_summaries["all"][machine]["strengthened_disallowed_seen"] += 1
          test_summaries[test_base][machine]["strengthened_disallowed_seen"] += 1
        else:
          test_summaries["all"][machine]["strengthened_allowed_seen"] += 1
          test_summaries["all"][machine]["strengthened_matched"] += 1
          test_summaries[test_base][machine]["strengthened_allowed_seen"] += 1

        if row["base"] == "UNSAT":
          base_matched = False
          base_disallowed_matched = False
          print(f"Unexpected weak test on {machine} under base: {test}")
          test_summaries["all"][machine]["base_disallowed_seen"] += 1
          test_summaries[test_base][machine]["base_disallowed_seen"] += 1
        else:
          test_summaries["all"][machine]["base_allowed_seen"] += 1
          test_summaries["all"][machine]["base_matched"] += 1
          test_summaries[test_base][machine]["base_allowed_seen"] += 1
      else:
        if row["base"] == "UNSAT":
          test_summaries["all"][machine]["base_matched"] += 1
        else:
          base_allowed_matched = False
          base_matched = False
        if row["strengthened"] == "UNSAT":
          test_summaries["all"][machine]["strengthened_matched"] += 1
        else:
          strengthened_allowed_matched = False
          strengthened_matched = False

    if base_matched:
      test_summaries["all"]["base_matched"] += 1
    if strengthened_matched:
      test_summaries["all"]["strengthened_matched"] += 1

    if row["base"] == "INSTANCE" and base_allowed_matched:
      test_summaries["all"]["base_allowed_matched"] += 1
    if row["base"] == "UNSAT" and base_disallowed_matched:
      test_summaries["all"]["base_disallowed_matched"] += 1

    if row["strengthened"] == "INSTANCE" and strengthened_allowed_matched:
      test_summaries["all"]["strengthened_allowed_matched"] += 1
    if row["strengthened"] == "UNSAT" and strengthened_disallowed_matched:
      test_summaries["all"]["strengthened_disallowed_matched"] += 1

  print(json.dumps(test_summaries, indent=2))

if __name__ == "__main__":
    main()

