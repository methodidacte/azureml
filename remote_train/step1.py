import argparse
import os
import numpy as np
import glob

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import joblib
import pickle

from azureml.core import Run
from azureml.core import Dataset
from utils import load_data

print("In step1.py")

parser = argparse.ArgumentParser("step1")

parser.add_argument("--input_data", type=str, help="input data")
parser.add_argument("--output_data", type=str, help="output data")
parser.add_argument('--regularization', type=float, dest='reg', default=0.5, help='regularization strength')

args = parser.parse_args()

print("Argument 1: %s" % args.input_data)
print("Argument 2: %s" % args.output_data)

run_context = Run.get_context()
ws = run_context.experiment.workspace
diabetes_input = run_context.input_datasets['diabetes_input']
diabetes = diabetes_input.to_pandas_dataframe()

target = "Y"
X = diabetes.drop(target, axis=1)
y = diabetes["Y"].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train a Ridge regression model with regularization strength of', args.reg)
model = Ridge(alpha=args.reg, solver="auto", random_state=42)
model.fit(X_train, y_train)

print('Predict the test set')
y_hat = model.predict(X_test)

# calculate score on the prediction
score = model.score(X_test, y_test)
print('Score is ', score)

diabetes['score'] = model.predict(X)

run_context.log('regularization strength', np.float(args.reg))
run_context.log('score', np.float(score))

if not (args.output_data is None):
    os.makedirs(args.output_data, exist_ok=True)
    print("%s created" % args.output_data)
    path = args.output_data + "/diabetesNext.csv"
    write_df = diabetes.to_csv(path, index=False)

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes_pipeline_model.pkl')
