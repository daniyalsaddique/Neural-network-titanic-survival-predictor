import streamlit as st
import joblib
import tensorflow as tf
import numpy as np

# Load artifacts
model = tf.keras.models.load_model('titanic_model.h5')
scaler = joblib.load('scaler.joblib')

# Streamlit UI
st.title("Titanic Survival Prediction 🚢")
st.markdown("Predict survival chances using passenger details")

# Input widgets
col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Gender", ["female", "male"])
    age = st.slider("Age", 0, 100, 25)
    
with col2:
    sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
    parch = st.slider("Parents/Children", 0, 6, 0)
    fare = st.slider("Fare", 0, 600, 50)

# Preprocess input
def prepare_features():
    return np.array([[pclass, 1 if sex == "male" else 0, age, sibsp, parch, fare]])

# Prediction
if st.button("Predict"):
    input_data = prepare_features()
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0][0]
    survival_prob = round(prediction * 100, 2)
    
    st.subheader(f"Survival Probability: {survival_prob}%")
    if survival_prob > 50:
        st.success("High chance of survival ✅")
    else:
        st.error("Low chance of survival ❌")
