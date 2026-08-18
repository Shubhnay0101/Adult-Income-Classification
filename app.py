import streamlit as st

st.set_page_config(
    page_title="Adult Census Income Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Adult Census Income Prediction")

st.write(
    "Machine Learning Classification Application"
)

st.info(
    "Upload a CSV file and select a trained model to generate income predictions."
)

# Upload CSV
st.subheader("📂 Upload Test Data")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    st.write("Preview of uploaded data:")

    import pandas as pd

    df = pd.read_csv(uploaded_file)

    st.dataframe(df.head())

    # Model Selection
st.subheader("Select Machine Learning Model")

model_options = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl"
}

selected_model = st.selectbox(
    "Choose a model",
    list(model_options.keys())
)

st.write("Selected Model:", selected_model)

import joblib
import os

# Load selected model
model_path = model_options[selected_model]

if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.success(f"{selected_model} loaded successfully!")
else:
    st.error(f"Model file not found: {model_path}")