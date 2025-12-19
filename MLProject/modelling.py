import mlflow
import mlflow.sklearn
import pandas as pd
import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data_preprocessing/credit_data_clean.csv")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()

# Create experiment
mlflow.set_experiment("Student Performance CI/CD")

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
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("random_state", args.random_state)
    mlflow.log_param("data_path", args.data_path)
    
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    
    # Log model
    mlflow.sklearn.log_model(
        model, 
        "model",
        signature=mlflow.models.infer_signature(X_train, model.predict(X_train))
    )
    
    # Save model locally for GitHub artifacts
    os.makedirs("../artifacts/model", exist_ok=True)
    joblib.dump(model, '../artifacts/model/model.pkl')