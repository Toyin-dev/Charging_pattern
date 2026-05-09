import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


TARGET_COLUMN = "Energy Consumed (kWh)"
MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "ev_charging_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.joblib"

RAW_COLUMNS = [
    "User ID",
    "Vehicle Model",
    "Charging Station ID",
    "Charging Station Location",
    "Charging Start Time",
    "Charging End Time",
    "Time of Day",
    "Day of Week",
    "Charger Type",
    "User Type",
    "Battery Capacity (kWh)",
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "Charging Cost (USD)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
    "Distance Driven (since last charge) (km)",
    "Temperature (°C)",
    "Vehicle Age (years)",
]

DROP_COLUMNS = [
    "User ID",
    "Charging Start Time",
    "Charging End Time",
]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in ["Charging Start Time", "Charging End Time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")

    if {"Charging Start Time", "Charging End Time"}.issubset(df.columns):
        df["calculated_duration_hours"] = (
            df["Charging End Time"] - df["Charging Start Time"]
        ).dt.total_seconds() / 3600
        df["start_hour"] = df["Charging Start Time"].dt.hour
        df["start_month"] = df["Charging Start Time"].dt.month
        df["is_weekend"] = df["Charging Start Time"].dt.dayofweek.isin([5, 6]).astype(float)

    if {"State of Charge (Start %)", "State of Charge (End %)"}.issubset(df.columns):
        df["soc_gain_percent"] = (
            df["State of Charge (End %)"] - df["State of Charge (Start %)"]
        )

    if {
        "Battery Capacity (kWh)",
        "State of Charge (Start %)",
        "State of Charge (End %)",
    }.issubset(df.columns):
        df["estimated_energy_from_soc_kwh"] = (
            df["Battery Capacity (kWh)"] * df["soc_gain_percent"] / 100
        )

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=5,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def calculate_rmse(y_true: pd.Series, predictions: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, predictions)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="ev_charging_patterns.csv",
        help="Path to the EV charging CSV file.",
    )
    args = parser.parse_args()

    csv_path = Path(args.data)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = load_data(csv_path)
    df = prepare_features(df)
    df = df.dropna(subset=[TARGET_COLUMN])

    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_rmse = calculate_rmse(y_test, baseline_predictions)
    baseline_r2 = r2_score(y_test, baseline_predictions)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = calculate_rmse(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(
        {
            "target": TARGET_COLUMN,
            "feature_columns": X.columns.tolist(),
            "raw_columns": RAW_COLUMNS,
            "metrics": {
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "baseline_r2": baseline_r2,
            },
        },
        METADATA_PATH,
    )

    print("Model training complete.")
    print(f"Saved model: {MODEL_PATH.resolve()}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2: {r2:.3f}")
    print(f"Baseline MAE: {baseline_mae:.3f}")
    print(f"Baseline RMSE: {baseline_rmse:.3f}")
    print(f"Baseline R2: {baseline_r2:.3f}")
    if r2 < 0.05:
        print(
            "Warning: model R2 is very low. The target appears weakly related "
            "to the available features in this dataset."
        )


if __name__ == "__main__":
    main()
