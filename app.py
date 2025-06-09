import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load('predictive_maintenance_model.pkl')

# Page setup
st.set_page_config(page_title="Predictive Maintenance", layout="centered")
st.title("🔧 Predictive Maintenance Dashboard")
st.markdown("Predict the likelihood of equipment failure using sensor inputs.")

st.divider()

# Sidebar - Sensor Inputs with Tooltips
st.sidebar.header("🔢 Sensor Inputs")

sensor_1 = st.sidebar.number_input(
    'Sensor 1', min_value=0.0, value=0.0,
    help="Temperature inside the machine (°C)"
)
sensor_2 = st.sidebar.number_input(
    'Sensor 2', min_value=0.0, value=0.0,
    help="Vibration intensity level"
)
sensor_3 = st.sidebar.number_input(
    'Sensor 3', min_value=0.0, value=0.0,
    help="Pressure in the hydraulic system (bar)"
)
sensor_4 = st.sidebar.number_input(
    'Sensor 4', min_value=0.0, value=0.0,
    help="Oil viscosity sensor (arbitrary units)"
)
sensor_5 = st.sidebar.number_input(
    'Sensor 5', min_value=0.0, value=0.0,
    help="RPM (rotations per minute) of the motor"
)

features = np.array([[sensor_1, sensor_2, sensor_3, sensor_4, sensor_5]])

# Predict button
if st.sidebar.button("🔍 Predict"):
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    st.subheader("🧠 Prediction Result")
    col1, col2 = st.columns(2)

    # Define alert threshold
    threshold = 0.6

    # Display detailed alert based on probability
    with col1:
        if probability > threshold:
            st.error("⚠️ ALERT: High failure risk detected. Recommend scheduling maintenance.")
        else:
            st.success("✅ Equipment operating within safe range.")

    # Add status badge and probability metric
    with col2:
        st.metric("Failure Probability", f"{probability:.2%}")
        if probability > threshold:
            st.markdown("**🔔 Status: At Risk – Action Needed**", unsafe_allow_html=True)
        else:
            st.markdown("**✅ Status: Stable**", unsafe_allow_html=True)

    st.write("**Probability Gauge:**")
    st.progress(min(int(probability * 100), 100))

    # Store prediction in session
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

    # Downloadable CSV
    results_df = pd.DataFrame([st.session_state.history[-1]])
    csv = results_df.to_csv(index=False)
    st.download_button("📄 Download This Prediction", data=csv, file_name="prediction_result.csv", mime='text/csv')

    st.divider()

    # Prediction history
    st.subheader("📈 Prediction History (This Session)")
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
    st.line_chart(history_df[['Failure Probability']])

# Info Section
with st.expander("ℹ️ About this App & How it Works"):
    st.markdown("""
    **🔧 What this app does:**
    - Uses live sensor inputs to predict equipment failure.
    - Built with a trained machine learning model (Random Forest).
    - Helps schedule maintenance before breakdowns happen.

    **💡 How to use it:**
    - Input 5 sensor readings on the left.
    - Click **Predict** to get a result.
    - Download predictions or track session history live.

    **🏭 Ideal for:**
    - Manufacturing plants
    - Industrial automation systems
    - Facility management teams
    """)

# Deployment instructions
with st.expander("🚀 Deploy This App"):
    st.markdown("""
    **To deploy online:**

    - Push `app.py` and `predictive_maintenance_model.pkl` to a GitHub repo.
    - Go to [streamlit.io/cloud](https://streamlit.io/cloud), sign in, and paste your repo URL.
    - Done. Your app is live and shareable.

    Or try [Hugging Face Spaces](https://huggingface.co/spaces) and choose **Streamlit** as your app type.
    """)

