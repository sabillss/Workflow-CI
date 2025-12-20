import mlflow
import pandas as pd
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="namadataset_preprocessing/student_performance_processed.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Create experiment
mlflow.set_experiment("Student Performance CI/CD")

# Enable autologging
mlflow.autolog()

# Load data
print(f"Loading data from: {args.data_path}")
df = pd.read_csv(args.data_path)

# Target column
target_column = 'G3'
X = df.drop(target_column, axis=1)
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state
)

# Start MLflow run
with mlflow.start_run(run_name="CI_Training_Run"):
    # Train model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Auto evaluate and log metrics
    mlflow.evaluate(model, X_test, y_test, targets=y_test, model_type="regressor")
    
    # Save model locally for GitHub artifacts
    os.makedirs("../artifacts/model", exist_ok=True)
    joblib.dump(model, '../artifacts/model/model.pkl')