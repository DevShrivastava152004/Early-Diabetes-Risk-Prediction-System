import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

# Load model
model = pickle.load(open("models/log_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

# ---------------- FUNCTIONS ---------------- #

def generate_explanation(glucose, bmi, age, dpf, bp):
    reasons = []

    if glucose >= 140:
        reasons.append("Elevated glucose levels detected.")
    elif glucose >= 110:
        reasons.append("Borderline glucose levels observed.")

    if bmi >= 30:
        reasons.append("BMI indicates obesity-related metabolic risk.")
    elif bmi >= 25:
        reasons.append("BMI indicates overweight category.")

    if age >= 50:
        reasons.append("Higher age increases diabetes susceptibility.")

    if dpf >= 0.5:
        reasons.append("Genetic risk factor is elevated.")

    if bp >= 140:
        reasons.append("High blood pressure may contribute to metabolic stress.")

    return reasons


def generate_recommendations(risk_level):
    recommendations = []

    if risk_level == "Low":
        recommendations += [
            "Maintain a balanced diet and regular physical activity.",
            "Monitor glucose levels annually."
        ]

    elif risk_level == "Mild":
        recommendations += [
            "Adopt a low refined-carb diet.",
            "Increase physical activity (150 min/week).",
            "Monitor fasting glucose every 3–6 months."
        ]

    elif risk_level == "High":
        recommendations += [
            "Consult a healthcare professional.",
            "Follow structured weight management.",
            "Reduce processed sugars.",
            "Monitor blood glucose monthly."
        ]

    elif risk_level == "Critical":
        recommendations += [
            "Seek immediate medical consultation.",
            "Undergo HbA1c and fasting glucose tests.",
            "Begin supervised lifestyle intervention."
        ]

    return recommendations


# ---------------- SIDEBAR ---------------- #

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Risk Prediction", "Model Info"])


# ---------------- RISK PREDICTION PAGE ---------------- #

if page == "Risk Prediction":

    st.title("🩺 Early Diabetes Risk Prediction System")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Enter Patient Details")

        preg = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 300, 120)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 28.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 1, 120, 30)

        predict = st.button("Predict Risk")

    if predict:

        with col2:

            glucose_bmi = glucose * bmi
            input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age, glucose_bmi]])
            input_scaled = scaler.transform(input_data)
            probability = model.predict_proba(input_scaled)[0][1]

            st.subheader("📊 Risk Assessment")
            st.metric("Risk Probability", f"{probability:.2%}")
            st.progress(float(probability))

            # Risk Segmentation
            if probability < 0.20:
                st.success("🟢 Low Risk")
                risk_level = "Low"
            elif probability < 0.40:
                st.info("🟡 Mild Risk")
                risk_level = "Mild"
            elif probability < 0.70:
                st.warning("🟠 High Risk")
                risk_level = "High"
            else:
                st.error("🔴 Critical Risk")
                risk_level = "Critical"

            confidence = abs(probability - 0.5) * 2

            st.markdown("---")
            st.subheader("🔎 Risk Explanation")

            reasons = generate_explanation(glucose, bmi, age, dpf, bp)

            if reasons:
                for r in reasons:
                    st.write("•", r)
            else:
                st.write("No major high-risk indicators detected.")

            st.markdown("---")
            st.subheader("💡 Recommended Actions")

            recommendations = generate_recommendations(risk_level)

            for rec in recommendations:
                st.write("•", rec)

            st.markdown("---")
            st.write("Confidence Score:", round(confidence, 2))

            # Report generation
            report = f"""
Early Diabetes Risk Assessment Report

Risk Probability: {probability:.2f}
Risk Category: {risk_level}
Confidence Score: {round(confidence,2)}
"""

            for rec in recommendations:
                report += f"- {rec}\n"

            st.download_button(
                "📄 Download Risk Report",
                report,
                file_name="diabetes_risk_report.txt"
            )


# ---------------- MODEL INFO PAGE ---------------- #

if page == "Model Info":

    st.title("📈 Model Performance Overview")

    st.write("Model: Logistic Regression")
    st.write("AUC Score: 0.81")
    st.write("Optimized Threshold: 0.24")
    st.write("Recall (Diabetic Class): 0.89")

    st.markdown("---")
    st.subheader("Key Features")

    st.write("• Glucose")
    st.write("• BMI")
    st.write("• Age")
    st.write("• Diabetes Pedigree Function")