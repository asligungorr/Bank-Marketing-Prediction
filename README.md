# 🏦 Bank Marketing Prediction

## 📊 Project Overview

This project is a machine learning application designed to predict whether a bank customer will subscribe to a term deposit based on historical marketing data. Using a trained **Random Forest Classifier**, the application provides real-time predictions through an intuitive **Streamlit** interface.

The primary objective is to assist marketing teams in improving campaign efficiency by identifying likely responders before initiating contact.

---

## 🚀 Key Features

- 🔮 **Live Prediction Interface** – Input customer details and receive instant predictions  
- 📈 **Model Performance Visualization** – Includes ROC curve, confusion matrix, and feature importance  
- ⚖️ **Class Imbalance Handling** – Uses techniques to manage skewed target classes  
- 🧠 **Trained ML Model** – Implements a Random Forest model with optimized hyperparameters  
- 🧪 **Reproducible Workflow** – All data processing and model training steps are documented

---

## 🧰 Tech Stack

| Layer        | Tools & Libraries                     |
|--------------|----------------------------------------|
| ML Model     | `scikit-learn` (RandomForestClassifier) |
| Web UI       | `Streamlit`                            |
| Data Handling| `pandas`, `numpy`                      |
| Visualization| `Plotly`, `matplotlib`, `seaborn`      |
| Serialization| `joblib`                               |
| Language     | `Python`                               |

---

## 🗂️ File Structure


``` bash
bank-marketing-prediction/
├── app.py # Streamlit web app for live predictions
├── project.ipynb # Jupyter notebook for EDA and model training
├── BSA_model.pkl # Trained Random Forest model (serialized)
├── bank-additional.csv # Dataset used for training and evaluation
├── requirements.txt # Python dependencies
├── pie_chart.png # Class imbalance visualization (target variable)
├── roc_curve.png # Model ROC curve
└── report_table.jpg # Classification metrics (precision, recall, f1) 
```
## 📁 Dataset Description

- **Source**: UCI Machine Learning Repository  
- **Size**: 41,188 records, 21 features  
- **Target Variable**: `y` – whether the client subscribed to a term deposit (`yes` / `no`)  
- **Type**: Tabular, classification  

---

## 📊 Model Summary

- **Algorithm**: Random Forest Classifier  
- **Accuracy**: 98.02%  
- **F1 Score**: 0.98  
- **AUC-ROC**: 1.00  
- **Optimization**: Grid search & manual tuning  

All model training, evaluation, and validation steps are fully documented in the Jupyter notebook.

---

## ▶️ How to Run the App

```bash
# Clone the repository
git clone https://github.com/asligungorr/Bank-Marketing-Prediction.git

# Navigate into the project directory
cd Bank-Marketing-Prediction

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py

## 🌐 Live Deployment

The application is currently deployed on **Streamlit Cloud**.  
If you would like to access the live demo, the deployment link can be provided upon request.
