Credit Card Fraud Detection:
A machine learning-powered web application to detect fraudulent credit card transactions in real time. Built using XGBoost, SMOTE for balancing, SHAP for model explainability, and Streamlit for the front-end.

Project Overview:
This project detects potentially fraudulent transactions using a trained classification model and displays predictions through an interactive Streamlit web app. The model is trained on anonymized credit card transaction data.

Features:
Fraud Prediction – Classifies transactions as fraudulent or legitimate.
SMOTE – Handles imbalanced datasets to improve accuracy.
XGBoost – A powerful tree-based algorithm for fast and accurate predictions.
Explainability – SHAP plots show why the model made a prediction.
Web App Interface – Simple and interactive UI built with Streamlit.

Tech Stack : XGBoost , SMOTE , SHAP , Streamlit , Python

File Structure:
├── fraud_detection.py - Core ML logic: training, prediction, SHAP
├── streamlit_app.py - Streamlit UI script
├── fraud_model.joblib - Saved XGBoost model
├── test_csv.csv - Sample CSV to test bulk predictions
├── requirements.txt - Python dependencies
└── README.md - Project documentation

How It Works:
1. Data is preprocessed and balanced using SMOTE.
2. An XGBoost model is trained and saved.
3. The model is loaded into a Streamlit app that:
   - Accepts user input (manual or CSV upload)
   - Predicts whether the transaction is fraudulent
   - Displays SHAP explanations for transparency

Installation:
bash
git clone https://github.com/Himas18/CreditCard_Fraud_Detection.git
cd CreditCard_Fraud_Detection
pip install -r requirements.txt

Usage: 
run the app - streamlit run streamlit_app.py

Demo:
Upload a CSV file or manually input values into the UI to see prediction results along with SHAP-based model explainability.

License:
This project is licensed under the MIT License. Feel free to use, modify, and share!
