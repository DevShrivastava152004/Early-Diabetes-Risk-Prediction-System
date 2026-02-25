import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")
st.markdown("<h2 style='color:#2E86C1;'>Health Risk Summary</h2>", unsafe_allow_html=True)

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
            percentage = round(probability * 100)
            st.metric("Estimated Risk Level", f"{percentage}%")
            st.progress(float(probability))

            # Risk Segmentation
            if probability < 0.20:
               risk_level= "Low"
               st.success("🟢 Low Risk\n\nYour health indicators look stable. Continue healthy habits.")
            elif probability < 0.40:
                risk_level = "Mild"
                st.info("🟡 Mild Risk\n\nThere are early warning signs. Improve diet and activity.")
            elif probability < 0.70:
                risk_level = "High"
                st.warning("🟠 High Risk\n\nMedical consultation is recommended.")
            else:
                risk_level = "Critical"
                st.error("🔴 Critical Risk\n\nPlease visit a healthcare professional immediately.")

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
            confidence_percent = round(confidence * 100)
            st.write(f"Reliability of Assessment: {confidence_percent}%")
            st.caption("Higher reliabilty means the system is more certain about this result. ")

            st.markdown("---")
            st.subheader("📝 Final Summary")

            if risk_level in ["Low"]:
                st.write("At this time, your health indicators look stable. Continue healthy habits.")
            elif risk_level in ["Mild"]:
                st.write("There are early signs of risk. Improving diet and exercise can help prevent diabetes.")
            elif risk_level in ["High"]:
                st.write("There are strong warning signs. Please consult a healthcare provider soon")
            else:
                st.write("Immediate medical evaluation is strongly recommended.")

            # Report generation
            report = f"""
Early Diabetes Risk Assessment Report

Risk Level: {percentage}%
Risk Category: {risk_level}
Reliability: {confidence_percent}%
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
