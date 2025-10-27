import pandas as pd
import argparse
import json

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("file_path", help="File to parse")

  args = parser.parse_args()
  df = pd.read_csv(args.file_path)

  machines = ["ampere", "hopper"]

  test_summaries = {}
  test_summaries["all"] = {"tests": 0, "relaxed_tests": 0, "ptx_allowed": 0, "ptx_mca_allowed": 0, "ptx_matched": 0, "ptx_mca_matched": 0, "ptx_allowed_matched": 0, "ptx_mca_allowed_matched": 0, "ptx_disallowed_matched": 0, "ptx_mca_disallowed_matched": 0}

  for machine in machines:
      test_summaries["all"][machine] = {"ptx_allowed_seen": 0, "ptx_disallowed_seen": 0, "ptx_mca_allowed_seen": 0, "ptx_mca_disallowed_seen": 0, "relaxed_seen": 0, "ptx_matched": 0, "ptx_mca_matched": 0}

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
      test_summaries[test_base] = {"tests": 0, "relaxed_tests": 0, "ptx_allowed": 0, "ptx_mca_allowed": 0, "ampere_seen": 0, "hopper_seen": 0}
      for machine in machines:
        test_summaries[test_base][machine] = {"ptx_allowed_seen": 0, "ptx_disallowed_seen": 0, "ptx_mca_allowed_seen": 0, "ptx_mca_disallowed_seen": 0, "relaxed_seen": 0}

    if "RELAXED" in test:
      test_summaries[test_base]["relaxed_tests"] += 1
      test_summaries["all"]["relaxed_tests"] += 1
    else:
      test_summaries[test_base]["tests"] += 1
      test_summaries["all"]["tests"] += 1

      if row["PTX"] == "allowed":
        test_summaries[test_base]["ptx_allowed"] += 1
        test_summaries["all"]["ptx_allowed"] += 1

      if row["PTX_MCA"] == "allowed":
        test_summaries[test_base]["ptx_mca_allowed"] += 1
        test_summaries["all"]["ptx_mca_allowed"] += 1

    ptx_matched = True
    ptx_allowed_matched = True
    ptx_disallowed_matched = True
    ptx_mca_matched = True
    ptx_mca_allowed_matched = True
    ptx_mca_disallowed_matched = True
    for machine in machines:
      if row[machine] == "seen":
        if "RELAXED" in test:
          test_summaries["all"][machine]["relaxed_seen"] += 1
          test_summaries[test_base][machine]["relaxed_seen"] += 1
        else:
          if row["PTX_MCA"] == "disallowed":
            ptx_mca_matched = False
            ptx_mca_disallowed_matched = False
            print(f"Unexpected weak test on {machine} under PTX_MCA: {test}")
            test_summaries["all"][machine]["ptx_mca_disallowed_seen"] += 1
            test_summaries[test_base][machine]["ptx_mca_disallowed_seen"] += 1
          else:
            test_summaries["all"][machine]["ptx_mca_allowed_seen"] += 1
            test_summaries["all"][machine]["ptx_mca_matched"] += 1
            test_summaries[test_base][machine]["ptx_mca_allowed_seen"] += 1

          if row["PTX"] == "disallowed":
            ptx_matched = False
            ptx_disallowed_matched = False
            print(f"Unexpected weak test on {machine} under PTX: {test}")
            test_summaries["all"][machine]["ptx_disallowed_seen"] += 1
            test_summaries[test_base][machine]["ptx_disallowed_seen"] += 1
          else:
            test_summaries["all"][machine]["ptx_allowed_seen"] += 1
            test_summaries["all"][machine]["ptx_matched"] += 1
            test_summaries[test_base][machine]["ptx_allowed_seen"] += 1
      #elif "RELAXED" in test:
        continue
        print(f"Not seen relaxed on {machine}: {test}")
      else:
        if row["PTX"] == "disallowed":
          test_summaries["all"][machine]["ptx_matched"] += 1
        else:
          ptx_allowed_matched = False
          ptx_matched = False
        if row["PTX_MCA"] == "disallowed":
          test_summaries["all"][machine]["ptx_mca_matched"] += 1
        else:
          ptx_mca_allowed_matched = False
          ptx_mca_matched = False
    if "RELAXED" not in test:
      if ptx_matched:
        test_summaries["all"]["ptx_matched"] += 1
      if ptx_mca_matched:
        test_summaries["all"]["ptx_mca_matched"] += 1

      if row["PTX"] == "allowed" and ptx_allowed_matched:
        test_summaries["all"]["ptx_allowed_matched"] += 1
      if row["PTX"] == "disallowed" and ptx_disallowed_matched:
        test_summaries["all"]["ptx_disallowed_matched"] += 1

      if row["PTX_MCA"] == "allowed" and ptx_mca_allowed_matched:
        test_summaries["all"]["ptx_mca_allowed_matched"] += 1
      if row["PTX_MCA"] == "disallowed" and ptx_mca_disallowed_matched:
        test_summaries["all"]["ptx_mca_disallowed_matched"] += 1

  print(json.dumps(test_summaries, indent=2))

if __name__ == "__main__":
    main()

