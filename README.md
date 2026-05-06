This data is a synthetic dataset
Linear regression model has low prediction
correlation was almost zero

# EV Charging Energy Prediction Deployment

This project deploys a regression model that predicts electric vehicle charging energy consumption in `kWh`.

## Project Files

- `train_model.py`: trains the regression model and saves the model artifacts.
- `fastapi_app.py`: exposes the trained model through a FastAPI prediction endpoint.
- `streamlit_app.py`: provides a simple web interface for predictions.
- `requirements.txt`: Python dependencies.
- `DEPLOYMENT_GUIDE.md`: step-by-step local deployment guide.

## Target

```text
Energy Consumed (kWh)
```

This is a regression target because the value is continuous.

## Train The Model

```powershell
python train_model.py --data ev_charging_patterns.csv
```

The script saves artifacts in:

```text
artifacts/
```

## Start FastAPI

```powershell
uvicorn fastapi_app:app --reload
```

API docs:

```text
http://127.0.0.1:8000/docs
```

## Start Streamlit

Open another terminal and run:

```powershell
streamlit run streamlit_app.py
```

Streamlit app:

```text
http://127.0.0.1:8501
```

## Model Note

The deployed pipeline works end to end, but the dataset has weak predictive signal for `Energy Consumed (kWh)`. The model performs only slightly better than a baseline average predictor, which is an important data science finding from the analysis.
