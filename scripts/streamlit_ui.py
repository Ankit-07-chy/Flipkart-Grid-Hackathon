import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pygeohash as pgh
from pathlib import Path

st.set_page_config(page_title="Traffic Demand Prediction", layout="wide")

# Load the trained model
model_path = Path(__file__).parent.parent / 'lgbm_model.pkl'
if not model_path.exists():
    st.error("Model file not found. Please train the model first.")
    st.stop()

with open(model_path, 'rb') as f:
    model = pickle.load(f)

def preprocess_input(time_str, geohash, road_type, lanes, large_vehicles, landmarks,
                     temperature, weather):
    """
    Preprocess user input to match model training features
    """
    # Extract hour and minute
    hour = int(time_str.split(':')[0])
    minute = int(time_str.split(':')[1])

    # Create cylindrical features for hour
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    # Quarter hour
    quarter_hour = minute // 15

    # Decode geohash
    try:
        latitude, longitude = pgh.decode(geohash)
    except:
        st.error("Invalid geohash format")
        return None

    # Encode categorical variables
    large_vehicles_encoded = 1 if large_vehicles == "Allowed" else 0
    landmarks_encoded = 1 if landmarks == "Yes" else 0

    # One-hot encoding for RoadType
    road_type_residential = 1 if road_type == "Residential" else 0
    road_type_street = 1 if road_type == "Street" else 0
    road_type_unknown = 1 if road_type == "Unknown" else 0

    # One-hot encoding for Weather
    weather_rainy = 1 if weather == "Rainy" else 0
    weather_snowy = 1 if weather == "Snowy" else 0
    weather_sunny = 1 if weather == "Sunny" else 0
    weather_unknown = 1 if weather == "Unknown" else 0

    # Create input DataFrame with all features in correct order
    input_data = pd.DataFrame({
        'NumberofLanes': [lanes],
        'LargeVehicles': [large_vehicles_encoded],
        'Landmarks': [landmarks_encoded],
        'Temperature': [temperature],
        'hour': [hour],
        'minute': [minute],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'quarter_hour': [quarter_hour],
        'latitude': [latitude],
        'longitude': [longitude],
        'RoadType_Residential': [road_type_residential],
        'RoadType_Street': [road_type_street],
        'RoadType_Unknown': [road_type_unknown],
        'Weather_Rainy': [weather_rainy],
        'Weather_Snowy': [weather_snowy],
        'Weather_Sunny': [weather_sunny],
        'Weather_Unknown': [weather_unknown]
    })

    return input_data

def main1():
    st.title("FlipKart Grid - Traffic Demand Prediction")
    st.markdown("Predict traffic demand at any location and time")

    # Create tabs for better organization
    tab1, tab2 = st.tabs(["Prediction", "About"])

    with tab1:
        st.subheader("Enter Location and Time Details")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Time & Location")
            time_str = st.text_input("Time (HH:MM format)", value="12:00", help="Enter time in HH:MM format (e.g., 14:30)")
            geohash = st.text_input("Geohash", value="qp03wm", help="Geographic hash code (e.g., qp03wm)")

        with col2:
            st.markdown("### Road & Vehicle Details")
            road_type = st.selectbox(
                "Road Type",
                options=["Residential", "Street", "Highway", "Unknown"],
                help="Type of road in the location"
            )
            lanes = st.slider(
                "Number of Lanes",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of lanes available"
            )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("### Environmental Conditions")
            temperature = st.slider(
                "Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=20.0,
                step=0.5,
                help="Temperature in Celsius"
            )
            weather = st.selectbox(
                "Weather",
                options=["Sunny", "Rainy", "Snowy", "Foggy", "Unknown"],
                help="Weather condition"
            )

        with col4:
            st.markdown("### Restrictions & Landmarks")
            large_vehicles = st.selectbox(
                "Large Vehicles",
                options=["Allowed", "Not Allowed"],
                help="Whether large vehicles are permitted"
            )
            landmarks = st.selectbox(
                "Landmarks Nearby",
                options=["Yes", "No"],
                help="Presence of landmarks in the area"
            )

        # Prediction button
        if st.button("🔮 Predict Demand", use_container_width=True):
            with st.spinner("Making prediction..."):
                # Preprocess input
                input_data = preprocess_input(
                    time_str, geohash, road_type, lanes,
                    large_vehicles, landmarks, temperature, weather
                )

                if input_data is not None:
                    try:
                        # Make prediction on log-scaled data
                        y_pred_log = model.predict(input_data)

                        # Inverse transform from log scale
                        predicted_demand = np.expm1(y_pred_log[0])

                        # Ensure prediction is within valid range
                        predicted_demand = np.clip(predicted_demand, 0, 1)

                        # Display results
                        st.success("Prediction successful!")

                        col_result1, col_result2, col_result3 = st.columns(3)

                        with col_result1:
                            st.metric(
                                "Predicted Demand",
                                f"{predicted_demand:.6f}",
                                help="Normalized demand (0-1 scale)"
                            )

                        with col_result2:
                            demand_percentage = predicted_demand * 100
                            st.metric(
                                "Demand %",
                                f"{demand_percentage:.2f}%"
                            )

                        with col_result3:
                            if predicted_demand < 0.3:
                                intensity = "Low 🟢"
                            elif predicted_demand < 0.6:
                                intensity = "Medium 🟡"
                            elif predicted_demand < 0.8:
                                intensity = "High 🟠"
                            else:
                                intensity = "Very High 🔴"
                            st.metric("Traffic Intensity", intensity)

                        # Show input details
                        with st.expander("Input Details & Feature Values"):
                            st.write("**Processed Features:**")
                            st.json({
                                "Location": {"Geohash": geohash, "Latitude": input_data['latitude'].values[0],
                                           "Longitude": input_data['longitude'].values[0]},
                                "Time": {"Hour": input_data['hour'].values[0], "Minute": input_data['minute'].values[0]},
                                "Road": {"Type": road_type, "Lanes": lanes},
                                "Environment": {"Temperature": temperature, "Weather": weather},
                                "Restrictions": {"LargeVehicles": large_vehicles, "Landmarks": landmarks}
                            })

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")

    with tab2:
        st.markdown("### About This Model")
        st.markdown("""
        **Traffic Demand Prediction System** uses Machine Learning to forecast traffic demand at specific locations and times.

        #### Model Details
        - **Algorithm:** LightGBM Regressor
        - **Features:** 17 engineered features
        - **Training Data:** 77,299 records
        - **Target Scale:** Normalized demand (0-1)

        #### Key Features
        - **Temporal Features:** Hour, minute, and cyclic encodings
        - **Geospatial Features:** Latitude, longitude (decoded from geohash)
        - **Categorical Features:** Road type, weather, vehicle restrictions, landmarks

        #### How It Works
        1. Input location (geohash) and time details
        2. Model extracts temporal and geographic patterns
        3. Combines with environmental conditions
        4. Predicts normalized demand value (0-1)

        #### Output Interpretation
        - **0.0-0.3:** Low traffic demand 🟢
        - **0.3-0.6:** Medium traffic demand 🟡
        - **0.6-0.8:** High traffic demand 🟠
        - **0.8-1.0:** Very high traffic demand 🔴
        """)

        st.markdown("---")
        st.markdown("**For more details:** Check README.md and notepad.txt in the project root")

if __name__ == "__main__":
    main1()