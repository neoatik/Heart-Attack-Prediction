# 🫀 Heart Attack Risk Prediction - End-to-End Machine Learning Project

## 📌 Overview
This project aims to predict **Heart Attack Risk** using **Machine Learning**. It follows a complete **end-to-end pipeline**, including **data ingestion, preprocessing, model training, and deployment** with a **Flask web app**.

## 🚀 Features
- **Exploratory Data Analysis (EDA)** to understand patterns in heart-related health data.
- **Machine Learning Model Training** with multiple algorithms.
- **Achieved an accuracy of 64.75%** with the best model.
- **Data Pipeline Components:**
  - **Data Ingestion:** Load and process raw data.
  - **Data Transformation:** Handle missing values, scaling, and encoding.
  - **Model Trainer:** Train and evaluate ML models.
  - **Predict Pipeline:** Takes the data from Flask web app and predicts Risk using model.pkl.

## 🏃‍♂️ How to Run the Project
### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt

python app.py
Go to: http://127.0.0.1:5000/