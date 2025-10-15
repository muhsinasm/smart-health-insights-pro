# app.py — Smart Health Insights Pro 💓
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 🎨 Page configuration
st.set_page_config(page_title="Smart Health Insights Pro", page_icon="💓", layout="wide")

# 🌟 Header Section
st.markdown("""
    <div style="text-align:center; padding:10px 0; background:linear-gradient(90deg,#ff758c,#ff7eb3); border-radius:10px;">
        <h1 style="color:white;">💓 Smart Health Insights Pro</h1>
        <p style="color:white;">Predict Heart Disease & Diabetes Risks Instantly</p>
    </div>
""", unsafe_allow_html=True)

# 📂 Load datasets
heart = pd.read_csv("heart.csv")
diabetes = pd.read_csv("diabetes.csv")

# 🧹 Data Preprocessing
if 'num' in heart.columns:
    heart.rename(columns={'num':'target'}, inplace=True)
if heart['sex'].dtype == 'object':
    heart['sex'] = heart['sex'].map({'Male':1, 'Female':0})

heart.fillna(heart.median(numeric_only=True), inplace=True)
diabetes.fillna(diabetes.median(numeric_only=True), inplace=True)

# 🧠 Model Training
Xh = heart[['age','sex','chol','trestbps']]
yh = (heart['target'] > 0).astype(int)

Xd = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','Age']]
yd = diabetes['Outcome']

scaler_h = StandardScaler()
scaler_d = StandardScaler()
Xh_scaled = scaler_h.fit_transform(Xh)
Xd_scaled = scaler_d.fit_transform(Xd)

heart_model = LogisticRegression(max_iter=1000).fit(Xh_scaled, yh)
diabetes_model = LogisticRegression(max_iter=1000).fit(Xd_scaled, yd)

# 🧍 Sidebar User Inputs
st.sidebar.markdown("### 🩺 Enter Your Health Details")
age = st.sidebar.slider("Age", 10, 100, 25)
sex = st.sidebar.radio("Sex", ["Male", "Female"])
chol = st.sidebar.slider("Cholesterol (mg/dL)", 100, 600, 200)
bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
glucose = st.sidebar.slider("Glucose Level", 50, 250, 100)
bmi = st.sidebar.slider("Body Mass Index (BMI)", 10.0, 60.0, 25.0)
preg = st.sidebar.slider("Pregnancies", 0, 15, 0)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin Level", 0, 900, 80)

# 📊 Layout: 2 Columns for Results
col1, col2 = st.columns(2)

sex_val = 1 if sex == "Male" else 0
heart_input = scaler_h.transform([[age, sex_val, chol, bp]])
diab_input = scaler_d.transform([[preg, glucose, bp, skin, insulin, bmi, age]])

heart_prob = heart_model.predict_proba(heart_input)[0][1] * 100
diab_prob = diabetes_model.predict_proba(diab_input)[0][1] * 100

# 🚀 Predict Button
if st.sidebar.button("🔍 Predict My Health"):
    with col1:
        st.markdown("### ❤️ Heart Disease Risk")
        st.metric(label="Risk Level", value=f"{heart_prob:.2f}%")
        if heart_prob > 70:
            st.error("🚨 High Risk! Consult a doctor soon.")
        elif heart_prob >= 40:
            st.warning("⚠️ Moderate Risk. Regular exercise & healthy food advised.")
        else:
            st.success("✅ Low Risk. Keep living healthy!")

    with col2:
        st.markdown("### 🩸 Diabetes Risk")
        st.metric(label="Risk Level", value=f"{diab_prob:.2f}%")
        if diab_prob > 70:
            st.error("🚨 High Risk! Consult a doctor soon.")
        elif diab_prob >= 40:
            st.warning("⚠️ Moderate Risk. Monitor glucose regularly.")
        else:
            st.success("✅ Low Risk. Keep up your good habits!")

    # 🌿 Personalized Health Tips
    st.markdown("""
        <div style="background-color:#e3f2fd; padding:15px; border-radius:10px;">
        <h4>💡 General Health Tips</h4>
        <ul>
            <li>Stay hydrated — drink at least 2L of water daily.</li>
            <li>Walk for 30 minutes each day.</li>
            <li>Eat fiber-rich foods (vegetables, fruits, whole grains).</li>
            <li>Get 7–8 hours of sleep regularly.</li>
        </ul>
        </div>
    """, unsafe_allow_html=True)

# 🧭 Dashboard Section
st.markdown("## 📈 Data Insights Dashboard")

tab1, tab2 = st.tabs(["💗 Heart Data", "🩸 Diabetes Data"])

with tab1:
    fig1 = px.histogram(heart, x="age", color=(heart["target"]>0).map({True:"Heart Disease",False:"Healthy"}), 
                        title="Heart Disease by Age", color_discrete_sequence=["#ff758c","#8bc34a"])
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.histogram(diabetes, x="Glucose", color=(diabetes["Outcome"]==1).map({True:"Diabetes",False:"No Diabetes"}), 
                        title="Diabetes by Glucose Level", color_discrete_sequence=["#ef5350","#81c784"])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2025 Smart Health Insights Pro | Built with ❤️ by Muzi")
