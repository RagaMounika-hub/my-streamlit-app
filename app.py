import streamlit as st
import pandas as pd
import joblib

# ✅ Load the trained model
model = joblib.load(r"C:\Users\rgukt\Downloads\Telegram Desktop\best_model.pkl")

st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Classification")
st.markdown("Predict whether an employee earns >50K or ≤50K.")

# ✅ Sidebar: Inputs matching TRAINING COLUMNS
st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
fnlwgt = st.sidebar.number_input("fnlwgt (final weight)", min_value=10000, max_value=1000000, value=50000)
educational_num = st.sidebar.slider("Educational Number", 1, 16, 10)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc",
    "Federal-gov", "Local-gov", "State-gov",
    "Without-pay", "Never-worked"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced",
    "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family",
    "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Mexico", "Philippines", "Germany",
    "Canada", "India", "England", "China", "Other"
])
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=5000, value=0)
hours_per_week = st.sidebar.slider("Hours per Week", 1, 99, 40)

# ✅ Build input row
input_data = pd.DataFrame({
    'age': [age],
    'fnlwgt': [fnlwgt],
    'educational-num': [educational_num],
    'workclass': [workclass],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'native-country': [native_country],
    'capital-gain': [capital_gain],
    'capital-loss': [capital_loss],
    'hours-per-week': [hours_per_week]
})

st.write("### 🔎 Input Data")
st.write(input_data)

if st.button("Predict Salary Class"):
    pred = model.predict(input_data)[0]
    st.success(f"✅ Prediction: {pred}")

# ✅ Batch prediction
st.markdown("---")
st.subheader("📂 Batch CSV Prediction")
file = st.file_uploader("Upload a CSV file", type="csv")

if file:
    batch = pd.read_csv(file)
    st.write("Preview:", batch.head())
    preds = model.predict(batch)
    batch['Prediction'] = preds
    st.write(batch.head())
    csv = batch.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "batch_predictions.csv", "text/csv")
