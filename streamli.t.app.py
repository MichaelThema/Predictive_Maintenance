import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('predictive_maintenance_model.pkl')

# Set page layout
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("Predict equipment failure based on live sensor data. Ideal for real-time decision making in manufacturing and engineering environments.")

st.divider()

# Sidebar: User Inputs
st.sidebar.header("🔢 Sensor Inputs")
sensor_1 = st.sidebar.number_input('Sensor 1', min_value=0.0, value=0.0)
sensor_2 = st.sidebar.number_input('Sensor 2', min_value=0.0, value=0.0)
sensor_3 = st.sidebar.number_input('Sensor 3', min_value=0.0, value=0.0)
sensor_4 = st.sidebar.number_input('Sensor 4', min_value=0.0, value=0.0)
sensor_5 = st.sidebar.number_input('Sensor 5', min_value=0.0, value=0.0)

# Prepare input features
features = np.array([[sensor_1, sensor_2, sensor_3, sensor_4, sensor_5]])

# Prediction button
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("🧠 Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.error("⚠️ Failure Likely")
        else:
            st.success("✅ No Failure Detected")
    with col2:
        st.metric("Failure Probability", f"{probability:.2%}")

    st.write("**Probability Gauge:**")
    st.progress(min(int(probability * 100), 100))

    # Store in session history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        'Sensor 1': sensor_1,
        'Sensor 2': sensor_2,
        'Sensor 3': sensor_3,
        'Sensor 4': sensor_4,
        'Sensor 5': sensor_5,
        'Failure Probability': probability,
        'Prediction': "Failure" if prediction == 1 else "No Failure"
    })

    # Download as CSV
    results_df = pd.DataFrame([st.session_state.history[-1]])
    csv = results_df.to_csv(index=False)
    st.download_button("📄 Download This Prediction", data=csv, file_name="prediction_result.csv", mime='text/csv')

    st.divider()

    # History Chart
    st.subheader("📈 Prediction History (This Session Only)")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    st.line_chart(history_df[['Failure Probability']])
