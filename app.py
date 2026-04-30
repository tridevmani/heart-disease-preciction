# ================== IMPORTS ==================
import sys, os, base64, datetime, re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import joblib
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

from src.predict import predict_user

# ================== CONFIG ==================
st.set_page_config(page_title="Cardiovascular Intelligence", layout="wide")

# ================== FIXED PATH (IMPORTANT) ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models/model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "models/scaler.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models/features.pkl"))

# ================== BACKGROUND ==================
def get_base64(img_path):
    with open(os.path.join(BASE_DIR, img_path), "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("assets/heart.jpg")

st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(rgba(15,23,42,0.6), rgba(15,23,42,0.7)),
                url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
}}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.title("Cardiovascular Risk Intelligence Platform")
st.caption("Real-time predictive analytics for clinical decision support")
st.divider()

# ================== LAYOUT ==================
left, right = st.columns([1.2, 1])

# ================== INPUT ==================
with left:
    st.subheader("Patient Details")

    name = st.text_input("Full Name")
    phone = st.text_input("Phone Number")

    st.subheader("Clinical Parameters")

    age = st.slider("Age", 20, 100, 50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["typical angina","atypical angina","non-anginal pain","asymptomatic"])
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 400, 200)

    fbs = st.selectbox("Fasting Blood Sugar", ["Yes", "No"])
    ecg = st.selectbox("ECG", ["normal","ST-T abnormality","left ventricular hypertrophy"])
    hr = st.slider("Max Heart Rate", 60, 220, 120)
    ex_angina = st.selectbox("Exercise Angina", ["Yes", "No"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

    slope = st.selectbox("Slope", ["upsloping","flat","downsloping"])
    ca = st.selectbox("CA", ["0","1","2","3"])
    thal = st.selectbox("Thal", ["normal","fixed defect","reversible defect"])

    user_input = [age, sex, cp, bp, chol,
                  fbs, ecg, hr, ex_angina,
                  oldpeak, slope, ca, thal]

# ================== VALIDATION ==================
valid = True

if not name.strip():
    valid = False

if not re.fullmatch(r"\d{10}", phone):
    valid = False

# ================== MODEL ==================
if valid:
    pred, prob, risk = predict_user(model, scaler, user_input, feature_names)
else:
    pred, prob, risk = 0, 0, "N/A"

# ================== OUTPUT ==================
with right:
    st.subheader("Prediction Output")

    if not valid:
        st.warning("Enter valid Name and 10-digit Phone Number to proceed")
    else:
        if prob > 0.7:
            color = "red"
            st.error("High Risk Detected")
        elif prob > 0.4:
            color = "orange"
            st.warning("Moderate Risk")
        else:
            color = "green"
            st.success("Low Risk")

        st.metric("Probability", f"{prob:.2f}")
        st.metric("Risk Level", risk)

        # ================== GRAPH ==================
        st.markdown("### Risk Distribution")

        fig, ax = plt.subplots(figsize=(4,1.5))
        ax.barh(["Risk"], [prob], color=color, height=0.25)
        ax.set_xlim(0,1)
        ax.axis('off')

        st.pyplot(fig)

        # ================== RECOMMENDATIONS ==================
        st.markdown("### Recommendations")

        if prob > 0.7:
            recommendation = "Immediate medical consultation is advised."
        elif prob > 0.4:
            recommendation = "Adopt healthy lifestyle and monitor regularly."
        else:
            recommendation = "Maintain current healthy lifestyle."

        st.write(recommendation)

        # ================== PDF ==================
        st.markdown("### Download Report")

        def create_pdf():
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer)
            styles = getSampleStyleSheet()

            content = []

            content.append(Paragraph("<b>Cardiovascular Risk Report</b>", styles['Title']))
            content.append(Spacer(1, 10))

            date = datetime.datetime.now().strftime("%d-%m-%Y")
            content.append(Paragraph(f"Date: {date}", styles['Normal']))
            content.append(Spacer(1, 10))

            content.append(Paragraph("<b>Patient Information</b>", styles['Heading2']))
            content.append(Paragraph(f"Name: {name}", styles['Normal']))
            content.append(Paragraph(f"Phone: {phone}", styles['Normal']))
            content.append(Spacer(1, 10))

            content.append(Paragraph("<b>Prediction Result</b>", styles['Heading2']))
            content.append(Paragraph(f"Probability: {prob:.2f}", styles['Normal']))

            if prob > 0.7:
                dot_color = "red"
            elif prob > 0.4:
                dot_color = "orange"
            else:
                dot_color = "green"

            content.append(Paragraph(
                f'Risk Level: <b><font color="{dot_color}">●</font> {risk}</b>',
                styles['Normal']
            ))

            content.append(Spacer(1, 10))

            content.append(Paragraph("<b>Recommendations</b>", styles['Heading2']))
            content.append(Paragraph(recommendation, styles['Normal']))

            doc.build(content)
            buffer.seek(0)
            return buffer

        pdf = create_pdf()

        st.download_button("Download PDF", pdf, "report.pdf")

# ================== FOOTER ==================
st.divider()
st.caption("AI-powered Clinical Decision Intelligence System")