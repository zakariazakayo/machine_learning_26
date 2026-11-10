import joblib
import streamlit as st

st.title("Household Poverty Predictor")

@st.cache_resource                  # load the 3 files only once
def load_files():
    return (joblib.load("lrmodel.pkl"),
            joblib.load("educ_encoder.pkl"),
            joblib.load("scaler.pkl"))
model, encoder, scaler = load_files()

area = st.radio("Area", ["Rural", "Urban"])
hh_size = st.number_input("Household size", 1, 30, 5)
head_educ = st.selectbox("Education of head", encoder.categories_[0])
land = st.number_input("Land owned (acres)", 0.0, 500.0, 1.0)

if st.button("Predict"):
    educ = encoder.transform([[head_educ]])[0, 0]         # 1. encode
    size, acres = scaler.transform([[hh_size, land]])[0]  # 2. scale
    X = [[int(area == "Urban"), size, educ, acres]]       # 3. order
    p = model.predict_proba(X)[0, 1]                      # 4. predict
    if p >= 0.5:                                          # 5. label
        st.error(f"Poor (chance of being poor: {p:.0%})")
    else:
        st.success(f"Non-poor (chance of being poor: {p:.0%})")
