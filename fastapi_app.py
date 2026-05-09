from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from train_model import METADATA_PATH, MODEL_PATH, prepare_features


app = FastAPI(
    title="EV Charging Energy Prediction API",
    description="Predicts energy consumed during an electric vehicle charging session.",
    version="1.0.0",
)


class EVChargingInput(BaseModel):
    charging_station_id: str = Field(default="Station_001", example="Station_001")
    vehicle_model: str = Field(example="Tesla Model 3")
    charging_station_location: str = Field(example="Urban")
    time_of_day: str = Field(example="Evening")
    day_of_week: str = Field(example="Monday")
    charger_type: str = Field(example="Level 2")
    user_type: str = Field(example="Commuter")
    battery_capacity_kwh: float = Field(example=75.0, gt=0)
    charging_duration_hours: float = Field(example=2.5, gt=0)
    charging_rate_kw: float = Field(example=22.0, gt=0)
    charging_cost_usd: float = Field(example=15.0, ge=0)
    state_of_charge_start_percent: float = Field(example=20.0, ge=0, le=100)
    state_of_charge_end_percent: float = Field(example=80.0, ge=0, le=100)
    distance_driven_since_last_charge_km: float = Field(example=120.0, ge=0)
    temperature_c: float = Field(example=28.0)
    vehicle_age_years: float = Field(example=3.0, ge=0)
    charging_start_time: Optional[str] = Field(default=None, example="2026-05-06 08:30:00")
    charging_end_time: Optional[str] = Field(default=None, example="2026-05-06 11:00:00")


def load_artifacts():
    if not Path(MODEL_PATH).exists() or not Path(METADATA_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail="Model artifacts are missing. Run: python train_model.py --data your_dataset.csv",
        )

    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def input_to_dataframe(payload: EVChargingInput) -> pd.DataFrame:
    row = {
        "Vehicle Model": payload.vehicle_model,
        "Charging Station ID": payload.charging_station_id,
        "Charging Station Location": payload.charging_station_location,
        "Time of Day": payload.time_of_day,
        "Day of Week": payload.day_of_week,
        "Charger Type": payload.charger_type,
        "User Type": payload.user_type,
        "Battery Capacity (kWh)": payload.battery_capacity_kwh,
        "Charging Duration (hours)": payload.charging_duration_hours,
        "Charging Rate (kW)": payload.charging_rate_kw,
        "Charging Cost (USD)": payload.charging_cost_usd,
        "State of Charge (Start %)": payload.state_of_charge_start_percent,
        "State of Charge (End %)": payload.state_of_charge_end_percent,
        "Distance Driven (since last charge) (km)": payload.distance_driven_since_last_charge_km,
        "Temperature (°C)": payload.temperature_c,
        "Vehicle Age (years)": payload.vehicle_age_years,
        "Charging Start Time": payload.charging_start_time,
        "Charging End Time": payload.charging_end_time,
        "Energy Consumed (kWh)": np.nan,
    }
    return pd.DataFrame([row])


@app.get("/")
def health_check():
    return {"status": "ok", "message": "EV Charging API is running"}


@app.get("/model-info")
def model_info():
    _, metadata = load_artifacts()
    return metadata


@app.post("/predict")
def predict(payload: EVChargingInput):
    model, metadata = load_artifacts()

    input_df = input_to_dataframe(payload)
    input_df = prepare_features(input_df)
    feature_columns = metadata["feature_columns"]
    input_features = input_df.reindex(columns=feature_columns)

    prediction = model.predict(input_features)[0]

    return {
        "predicted_energy_consumed_kwh": round(float(prediction), 3),
        "target": metadata["target"],
    }
