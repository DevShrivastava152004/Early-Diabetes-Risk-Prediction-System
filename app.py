import streamlit as st
import numpy as np
import pickle


# Load model
model = pickle.load(open("models/log_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))


st.title("Early Diabetes Risk Prediction System")
st.header("Enter Patient Details")

preg = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)

glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)

bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)

skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)

insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80, step=1)

bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=28.0, step=0.1)

dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)

age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

if(st.button("Predict Risk")):

    # Create interaction feature
    glucose_bmi = glucose * bmi

    # ADD INTERACTION FEATURE
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age, glucose_bmi]])

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    threshold = 0.24

    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")
    st.write(f"Risk Probability: {probability: 2f}")

    if prediction == 1:
        st.error("High Risk Of Diabetes")
    else:
        st.success("Low Risk of Diabetes")

        st.write("Confidence Score:", round(abs(probability - 0.5) * 2,2))

