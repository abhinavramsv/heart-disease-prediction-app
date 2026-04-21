import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", page_icon="🫀", layout="centered")

st.title("🫀 Heart Disease Prediction App")
st.write("Enter the patient's vitals below to predict the likelihood of heart disease.")

@st.cache_resource
def load_assets():
    model = joblib.load("best_heart_disease_model.pkl")
    scaler = joblib.load("standard_scaler.pkl")
    if os.path.exists("model_columns.pkl"):
        cols = joblib.load("model_columns.pkl")
    else:
        cols = None
    return model, scaler, cols

try:
    model, scaler, model_columns = load_assets()
except Exception as e:
    st.error("Model assets not found! Please ensure you have run the training pipeline first.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[1, 0], format_func=lambda x: "True" if x==1 else "False")
    restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Major Vessels Colored by Fluoroscopy (ca)", options=[0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])

if st.button("Predict Heart Disease", use_container_width=True):
    input_data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }

    input_df = pd.DataFrame([input_data])

    input_df['ca_thal_risk'] = input_df['ca'] * input_df['thal']
    input_df['ca_high']      = (input_df['ca'] >= 2).astype(int)
    input_df['oldpeak_ca']   = input_df['oldpeak'] * input_df['ca']
    input_df['thalach_age']  = input_df['thalach'] / input_df['age']

    categorical_features = ['cp', 'restecg', 'slope', 'ca', 'thal']
    input_df = pd.get_dummies(input_df, columns=categorical_features, drop_first=True)

    if model_columns is not None:
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

    scaled_input = scaler.transform(input_df)

    proba = model.predict_proba(scaled_input)[0]
    heart_disease_prob = proba[1] * 100
    healthy_prob       = proba[0] * 100

    BOOST     = 0.15
    THRESHOLD = 0.35
    risk_score      = proba[1]
    override_reason = None

    if ca >= 2 and oldpeak >= 1.5:
        risk_score += BOOST
        override_reason = (
            f"{ca} major vessels blocked + ST depression {oldpeak} mm — "
            "multi-vessel disease with ischaemia (score boosted by +15%)."
        )
    elif exang == 1 and thal >= 2 and ca >= 1:
        risk_score += BOOST
        override_reason = (
            f"Exercise-induced angina + thalassemia type {thal} + {ca} blocked vessel(s) — "
            "triple risk marker combination (score boosted by +15%)."
        )

    risk_score = min(risk_score, 1.0)
    high_risk  = risk_score >= THRESHOLD

    st.markdown("---")
    if high_risk:
        st.error("\u26a0\ufe0f **Prediction: High Risk of Heart Disease**")
        st.write(
            f"Model probability: **{heart_disease_prob:.1f}%** "
            + (f"(boosted to **{risk_score*100:.1f}%** by clinical rule)" if override_reason else "")
        )
        if override_reason:
            st.warning("\U0001f6a8 **Clinical Rule Boost Applied**\n\n" + override_reason)
        elif heart_disease_prob < 50:
            st.warning("\u26a1 Borderline prediction (35\u201350%). Clinical review strongly advised.")
    else:
        st.success("\u2705 **Prediction: Low Risk of Heart Disease**")
        st.write(f"Model probability of being healthy: **{healthy_prob:.1f}%**")
    st.caption(
        f"Model: Random Forest  |  Threshold: \u226535%  "
        f"|  Raw HD Prob: {heart_disease_prob:.1f}%  "
        f"|  Effective Score: {risk_score*100:.1f}%"
        + ("  |  Rule boost: ACTIVE" if override_reason else "")
    )
