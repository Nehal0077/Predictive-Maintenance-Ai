import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="⚙️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("predictive_maintenance_model.pkl")

model = load_model()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #06111f, #0b2545);
    color: #eef6ff;
}
h1, h2, h3, p, label {
    color: #eef6ff !important;
}
div.stButton > button {
    background: #168cff;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background: #00c8ff;
    color: #06111f;
}
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Predictive Maintenance AI")
st.caption("Machine Learning-Based Failure Risk Prediction System")

st.divider()
st.subheader("Machine Sensor Parameters")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    air_temp = st.number_input(
        "Air Temperature (K)", 295.0, 305.0, 300.0, step=0.1
    )
    process_temp = st.number_input(
        "Process Temperature (K)", 305.0, 315.0, 310.0, step=0.1
    )
    speed = st.number_input(
        "Rotational Speed (rpm)", 1000, 3000, 1500, step=10
    )

with col2:
    torque = st.number_input(
        "Torque (Nm)", 3.0, 80.0, 40.0, step=0.5
    )
    tool_wear = st.number_input(
        "Tool Wear (min)", 0, 300, 100, step=5
    )

if st.button("🔍 Predict Machine Condition", use_container_width=True):
    car = pd.DataFrame([{
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Type_L": int(machine_type == "L"),
        "Type_M": int(machine_type == "M")
    }])

    prediction = model.predict(car)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Potential Machine Failure Detected")
        st.write("Arrange a machine inspection and verify sensor readings.")
    else:
        st.success("✅ No Failure Predicted")
        st.write("The model did not predict a failure for these inputs.")

    st.caption(
        "This is a machine-learning prediction, not a guarantee of machine safety."
    )

st.divider()
st.subheader("Model Performance — Test Dataset")

m1, m2, m3 = st.columns(3)
m1.metric("Algorithm", "Gradient Boosting")
m2.metric("Accuracy", "98.60%")
m3.metric("Failure Recall", "67.65%")

st.caption(
    "Academic prototype | Python • Pandas • Scikit-learn • Streamlit"
)
