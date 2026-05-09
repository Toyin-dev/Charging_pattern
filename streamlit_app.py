import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/predict"


st.set_page_config(
    page_title="EV Charging Predictor",
    page_icon="EV",
    layout="centered",
)

st.title("EV Charging Energy Predictor")

with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        charging_station_id = st.text_input("Charging Station ID", "Station_001")
        vehicle_model = st.text_input("Vehicle Model", "Tesla Model 3")
        charging_station_location = st.text_input("Charging Station Location", "Urban")
        time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
        day_of_week = st.selectbox(
            "Day of Week",
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        )
        charger_type = st.selectbox("Charger Type", ["Level 1", "Level 2", "DC Fast Charger"])
        user_type = st.selectbox("User Type", ["Commuter", "Casual Driver", "Long-Distance Traveler"])
        battery_capacity_kwh = st.number_input("Battery Capacity (kWh)", min_value=1.0, value=75.0)
        vehicle_age_years = st.number_input("Vehicle Age (years)", min_value=0.0, value=3.0)

    with right:
        charging_duration_hours = st.number_input(
            "Charging Duration (hours)", min_value=0.1, value=2.5
        )
        charging_rate_kw = st.number_input("Charging Rate (kW)", min_value=0.1, value=22.0)
        charging_cost_usd = st.number_input("Charging Cost (USD)", min_value=0.0, value=15.0)
        state_of_charge_start_percent = st.slider("State of Charge Start (%)", 0.0, 100.0, 20.0)
        state_of_charge_end_percent = st.slider("State of Charge End (%)", 0.0, 100.0, 80.0)
        distance_driven_since_last_charge_km = st.number_input(
            "Distance Driven Since Last Charge (km)", min_value=0.0, value=120.0
        )
        temperature_c = st.number_input("Temperature (C)", value=28.0)

    use_time = st.checkbox("Include charging start and end time", value=False)
    charging_start_time = None
    charging_end_time = None
    if use_time:
        time_left, time_right = st.columns(2)
        with time_left:
            charging_start_time = st.text_input("Charging Start Time", "2026-05-06 08:30:00")
        with time_right:
            charging_end_time = st.text_input("Charging End Time", "2026-05-06 11:00:00")

    submitted = st.form_submit_button("Predict Energy Consumed")


if submitted:
    payload = {
        "charging_station_id": charging_station_id,
        "vehicle_model": vehicle_model,
        "charging_station_location": charging_station_location,
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "charger_type": charger_type,
        "user_type": user_type,
        "battery_capacity_kwh": battery_capacity_kwh,
        "charging_duration_hours": charging_duration_hours,
        "charging_rate_kw": charging_rate_kw,
        "charging_cost_usd": charging_cost_usd,
        "state_of_charge_start_percent": state_of_charge_start_percent,
        "state_of_charge_end_percent": state_of_charge_end_percent,
        "distance_driven_since_last_charge_km": distance_driven_since_last_charge_km,
        "temperature_c": temperature_c,
        "vehicle_age_years": vehicle_age_years,
        "charging_start_time": charging_start_time,
        "charging_end_time": charging_end_time,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=15)
        response.raise_for_status()
        result = response.json()
        st.metric(
            "Predicted Energy Consumed",
            f"{result['predicted_energy_consumed_kwh']} kWh",
        )
    except requests.exceptions.RequestException as exc:
        st.error(
            "Could not reach the FastAPI service. Start it with: "
            "uvicorn fastapi_app:app --reload"
        )
        st.exception(exc)
