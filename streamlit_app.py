import streamlit as st
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Load model and encoders
model_data = joblib.load("fraud_model.joblib")  
model = model_data['fraud_model']
label_encoders = model_data['encoders']

st.title("Credit Card Fraud Detection System")

option = st.sidebar.selectbox("Choose Action", ["Single Prediction", "Bulk Predictions (CSV)"])

shap.initjs()

def clear_saved_images():
    for file in os.listdir():
        if file.endswith(".png"):
            os.remove(file)

def plot_shap_summary(shap_values, data):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(shap_values, data, plot_type="bar", show=False)
    st.pyplot(fig)
    plt.close(fig)
    clear_saved_images()

def plot_shap_waterfall(shap_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(shap_values[0], show=False)
    st.pyplot(fig)
    plt.close(fig)
    clear_saved_images()

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)

if option == "Single Prediction":
    st.sidebar.header("Transaction Details")

    def get_user_input():
        amount = st.sidebar.number_input("Transaction Amount", value=1000.0, step=10.0)
        category = st.sidebar.selectbox("Category", label_encoders['category'].classes_)
        location = st.sidebar.selectbox("Location", label_encoders['location'].classes_)
        device = st.sidebar.selectbox("Device", label_encoders['device'].classes_)
        is_international = st.sidebar.selectbox("International Transaction?", [0, 1])
        is_weekend = st.sidebar.selectbox("Weekend Transaction?", [0, 1])
        user_transaction_freq = st.sidebar.number_input("User Transaction Frequency", value=1, step=1)
        merchant_transaction_freq = st.sidebar.number_input("Merchant Transaction Frequency", value=1, step=1)

        input_data = pd.DataFrame({
            'amount': [amount],
            'category': [category],
            'location': [location],
            'device': [device],
            'is_international': [is_international],
            'is_weekend': [is_weekend],
            'user_transaction_freq': [user_transaction_freq],
            'merchant_transaction_freq': [merchant_transaction_freq],
        })
        return input_data

    input_data = get_user_input()

    for col in ['category', 'location', 'device']:
        input_data[col] = label_encoders[col].transform(input_data[col])

    input_data = input_data.astype({
        'amount': 'float',
        'is_international': 'int',
        'is_weekend': 'int',
        'user_transaction_freq': 'int',
        'merchant_transaction_freq': 'int'
    })

    if st.button("Predict Fraud"):
        prediction_proba = model.predict_proba(input_data)[:, 1]
        prediction = (prediction_proba >= 0.3).astype(int)

        st.subheader("Prediction Results")
        st.write(f"Fraud Probability: {prediction_proba[0]:.2f}")
        st.write("Prediction: Fraud" if prediction[0] == 1 else "Prediction: Not Fraud")

        explainer = get_explainer(model)
        shap_values_single = explainer(input_data.iloc[0:1])
        plot_shap_waterfall(shap_values_single)

elif option == "Bulk Predictions (CSV)":
    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        required_cols = ['amount', 'category', 'location', 'device', 'is_international', 'is_weekend', 'user_transaction_freq', 'merchant_transaction_freq']

        if not all(col in data.columns for col in required_cols):
            st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
        else:
            for col in ['category', 'location', 'device']:
                if col in data.columns:
                    data[col] = label_encoders[col].transform(data[col])

            data = data.astype({
                'amount': 'float',
                'is_international': 'int',
                'is_weekend': 'int',
                'user_transaction_freq': 'int',
                'merchant_transaction_freq': 'int'
            })

            predictions_proba = model.predict_proba(data)[:, 1]
            predictions = (predictions_proba >= 0.3).astype(int)
            data['Fraud Probability'] = predictions_proba
            data['Prediction'] = ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]

            st.subheader("Bulk Prediction Results")
            st.write(data.head())

            st.download_button(
                label="Download Results as CSV",
                data=data.to_csv(index=False),
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

            sample_size = min(50, len(data))
            sample_data = data.sample(n=sample_size, random_state=42)

            feature_columns = ['amount', 'category', 'location', 'device', 'is_international', 'is_weekend', 'user_transaction_freq', 'merchant_transaction_freq']
            sample_data = sample_data[feature_columns]

            explainer = get_explainer(model)
            shap_values_sample = explainer(sample_data)

            st.subheader("SHAP Visualization (Bulk)")
            plot_shap_summary(shap_values_sample, sample_data)
