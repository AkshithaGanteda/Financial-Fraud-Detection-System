💳 Financial Fraud Detection System

An AI-powered Financial Fraud Detection System built using Machine Learning and Streamlit to identify fraudulent transactions, perform risk analysis, and visualize fraud patterns through an interactive dashboard.

📌 Overview

This project is an end-to-end Machine Learning-based Financial Fraud Detection System developed using Python and Scikit-Learn.

The system predicts whether a financial transaction is fraudulent or legitimate using a Random Forest Classifier trained on highly imbalanced transaction data. The application provides real-time predictions, bulk transaction analysis, and interactive visualizations through a user-friendly Streamlit dashboard.

🚀 Features
Fraud Transaction Prediction
Manual Transaction Prediction
Bulk CSV Prediction
Machine Learning Model using Random Forest
SMOTE for Handling Imbalanced Datasets
Interactive Dashboard Analytics
Model Performance Comparison
Fraud Detection Rate Analysis
Data Visualization and Reporting
Real-Time Risk Assessment
Downloadable Prediction Results
🛠️ Technologies Used
🖥️ Programming & Data Processing
Python
Pandas
NumPy
Joblib
🤖 Machine Learning
Scikit-Learn
Random Forest Classifier
SMOTE (Synthetic Minority Oversampling Technique)
📊 Visualization & Interface
Streamlit
Matplotlib
Seaborn
Plotly
📂 Project Structure
financial-fraud-detection/
│
├── app/
│   └── app.py
│
├── models/
│   └── fraud_detection_model.pkl
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   ├── EDA.ipynb
│   └── model_comparison.ipynb
│
├── src/
│
├── requirements.txt
│
└── README.md
📊 Dataset Information
Credit Card Fraud Detection Dataset
Total Transactions: 284,807
Fraudulent Transactions: 492
Features: 30
Dataset Type: Highly Imbalanced
Dataset Source

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

⚙️ Machine Learning Workflow
Data Collection
Exploratory Data Analysis (EDA)
Data Preprocessing
Train-Test Split
SMOTE Oversampling
Random Forest Model Training
Model Evaluation
Streamlit Application Development
Deployment
📈 Model Evaluation Metrics

The model was evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix

The model prioritizes Recall to minimize the chances of missing fraudulent transactions, which can lead to significant financial losses.

🧠 Why SMOTE?

The dataset contains very few fraudulent transactions compared to legitimate transactions.

SMOTE (Synthetic Minority Oversampling Technique) was used to generate synthetic fraud samples and balance the dataset, resulting in improved fraud detection performance.

🖥️ Dashboard Modules
🏠 Dashboard
Dataset Overview
Model Performance Metrics
Fraud Analytics Overview
Transaction Distribution
Fraud Detection Rate
📝 Manual Prediction
Enter transaction details manually
Real-time fraud prediction
Risk score analysis
📂 CSV Prediction
Upload transaction datasets
Bulk fraud detection
Download prediction results
🤖 Model Comparison
Compare Machine Learning models
Visualize model performance
ℹ️ About
Project information
Technologies used
Future enhancements
▶️ Running the Project
Step 1: Clone the Repository
git clone <repository-url>
Step 2: Install Dependencies
pip install -r requirements.txt
Step 3: Run the Application
cd app
streamlit run app.py
🌐 Deployment

The application can be deployed using Streamlit Community Cloud to provide public access and real-time fraud detection capabilities.

💡 Future Enhancements
Real-Time Banking Integration
Email & SMS Fraud Alerts
Deep Learning-Based Detection Models
Live Monitoring Dashboard
REST API Integration
Explainable AI (XAI)
Cloud-Based Deployment
User Authentication & Role Management
👨‍💻 Author

Akshitha Ganteda

Financial Fraud Detection System using Machine Learning and Streamlit.