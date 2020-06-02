import argparse
import os
from azureml.core import Run

print("In step2.py")

parser = argparse.ArgumentParser("step2")

parser.add_argument("--old_score", type=float, help="Old score of the model")
parser.add_argument("--final_data", type=str, help="Final data")

#step2: error: unrecognized arguments: --output /mnt/batch/tasks/shared/LS_root/jobs/sandboxaml/azureml/0f5d7b6d-b1ce-49c7-abd7-fa4c45c52660/mounts/workspaceblobstore/azureml/0f5d7b6d-b1ce-49c7-abd7-fa4c45c52660/final_data

args = parser.parse_args()

print("Old score: %s" % args.old_score)
print("Output : %s" % args.final_data)

run = Run.get_context()
processed_data = run.input_datasets["processed_data"]
final_data = processed_data.to_pandas_dataframe()

#compare to new score


#register if better


final_data["End"] = 2

if not (args.final_data is None):
    os.makedirs(args.final_data, exist_ok=True)
    print("%s created" % args.final_data)
    path = args.final_data + "/diabetesEnd.csv"
    write_df = final_data.to_csv(path, index=False)
