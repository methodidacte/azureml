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

# get hold of the current run
run = Run.get_context()
exp = run.experiment
ws = run.experiment.workspace

parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float, dest='reg', default=0.5, help='regularization strength')
args = parser.parse_args()

# load train and test set into numpy arrays
diabetes_dataset = Dataset.get_by_name(ws, name='diabetes')
diabetes = diabetes_dataset.to_pandas_dataframe().drop("Path", axis=1)
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

run.log('regularization strength', np.float(args.reg))
run.log('score', np.float(score))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes_reg_remote_model.pkl')
