# EV Charging Deployment Guide

This project predicts `Energy Consumed (kWh)` for an EV charging session.

## Files

- `train_model.py`: trains the model and saves artifacts.
- `fastapi_app.py`: serves predictions through FastAPI.
- `streamlit_app.py`: provides a simple web interface.
- `requirements.txt`: packages needed to run the project.

## 1. Install Packages

```powershell
pip install -r requirements.txt
```

## 2. Train And Save The Model

Put your dataset CSV in this folder, then run:

```powershell
python train_model.py --data ev_charging_patterns.csv
```

This creates:

```text
artifacts/ev_charging_model.joblib
artifacts/metadata.joblib
```

The saved pipeline contains:

- missing-value imputers
- `MinMaxScaler` for numeric features
- one-hot encoding for categorical features
- `RandomForestRegressor`

## 3. Start FastAPI

```powershell
uvicorn fastapi_app:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## 4. Start Streamlit

Open another terminal in the same folder and run:

```powershell
streamlit run streamlit_app.py
```

Streamlit will open a browser page where you can enter EV charging details and get a prediction.

## Notes

This is a regression deployment because the target is numeric:

```text
Energy Consumed (kWh)
```

If you want classification instead, create a category target such as `Low`, `Medium`, and `High` energy consumption before training.
