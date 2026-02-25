import streamlit as st
import numpy as np
import pickle

def generate_explanation(glucose, bmi, age, dpf, bp):
    reasons = []

    if glucose >= 140:
        reasons.append("Elevated glucose levels detected.")

    elif glucose >= 110:
        reasons.append("Borderline glucose levels observed.")

    if bmi >= 30:
        reasons.append("BMI indicated obesity-related metabolic risk.")

    elif bmi >= 25:
        reasons.append("BMI indicated overweight category.")

    if age >= 50:
        reasons.append("Higher age increases diabetes susceptibility.")

    if dpf >= 0.5:
        reasons.append("Genetic risk factor (Diabetes Pedigree Function) is elevated.")

    if bp >= 140:
        reasons.append("High blood pressure may contribute to metabolic stress.")

    return reasons

def generate_recommendations(risk_level):
    recommendations = []

    if risk_level == "Low":
        recommendations.append("Maintain a balanced diet and regular physical activity.")
        recommendations.append("Monitor glucose levels annually.")

    elif risk_level == "Mild":
        recommendations.append("Adopt a low-refined-carb-diet.")
        recommendations.append("Increase Physical activity (atleast 150 min/week).")
        recommendations.append("Monitor fasting glucose every 3-6 months.")

    elif risk_level == "High":
        recommendations.append("Consult a healthcare professional for evaluation.")
        recommendations.append("Follow a structured weight management plan.")
        recommendations.append("Reduce refined sugars and processed foods.")
        recommendations.append("Monitor blood glucose monthly.")

    elif risk_level == "Critical":
        recommendations.append("Seek immediate medical consultation.")
        recommendations.append("Undergo detailed blood tests (Hba1c, fasting glucose).")
        recommendations.append("Begin supervised lifestyle intervention.")
        recommendations.append("Follow strict dietary control and physical monitoring. ")

    return recommendations


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

    st.subheader("Preduction Result")
    st.write(f"Risk Probability: {probability:.2f}")

    # Risk Segmentation

    if probability < 0.20:
        st.success("Low RisK catergory")
        risk_level = "Low"
    elif probability< 0.40:
        st.info("Mild Risk Category")
        risk_level = "Mild"
    elif probability < 0.70:
        st.warning("High Risk Category")
        risk_level = "High"
    else:
        st.error("Critical Risk Category")
        risk_level = "Critical"


    reasons = generate_explanation(glucose, bmi, age, dpf, bp)
    st.subheader("Risk Explanation")

    if reasons:
        for reason in reasons:
            st.write("*", reason)
    else:
        st.write("No major high-risk indicators detected.")

    st.subheader("Recommended Actions")

    recommendations = generate_recommendations(risk_level)

    for rec in recommendations:
        st.write("•", rec)

        # Confidence Score
        confidence = abs(probability - 0.5) * 2
        st.write("Confidence Score:", round(confidence,2))

