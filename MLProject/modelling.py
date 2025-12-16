import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import mlflow
import mlflow.sklearn

print("=== modelling.py START ===")

def main():
    # 1) tracking lokal (biar jelas nyimpen di folder mlruns)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("student_performance_g3")

    # 2) Load dataset preprocessing (BUKAN RAW)
    data_path = "namadataset_preprocessing/student_performance_processed.csv"
    df = pd.read_csv(data_path)

    target_col = "G3"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' tidak ditemukan!")

    # 3) pastikan boolean jadi 0/1 (biar aman untuk sklearn)
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # kalau masih ada object, stop (berarti preprocessing belum bersih)
    if (X.dtypes == "object").any():
        obj_cols = X.columns[X.dtypes == "object"].tolist()
        raise ValueError(f"Masih ada kolom object: {obj_cols}")

    # 4) split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) autolog (WAJIB Basic)
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="rf_regression_g3"):
        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        # FIX: scikit-learn kamu nggak support squared=False
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        r2 = r2_score(y_test, preds)

        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2)


if __name__ == "__main__":
    main()
