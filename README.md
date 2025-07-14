# Bank-Marketing-Prediction
This is a machine learning project to predict whether a customer will subscribe to a term deposit, based on bank marketing data. The app is built with Streamlit and uses a trained Random Forest model under the hood.

## About the Project

The dataset comes from a Portuguese bank’s direct marketing campaigns. The goal is to help identify potential customers before contacting them, improving campaign efficiency.

## Features

- Predicts subscription using real-time user input
- Interactive UI built with Streamlit
- Handles imbalanced data
- Visualizes model performance (ROC curve, confusion matrix, feature importance)

## Tech Stack

- Python  
- scikit-learn (RandomForestClassifier)  
- Streamlit  
- pandas, numpy  
- Plotly  
- joblib

## File Structure
``` bank-marketing-prediction/
├── app.py # Main Streamlit app
├── project.ipynb # Notebook for model training & evaluation
├── BSA_model.pkl # Trained model
├── bank-additional.csv # Dataset
├── requirements.txt # Dependencies
├── pie_chart.png # Class distribution
├── roc_curve.png # ROC curve
└── report_table.jpg # Classification report 
```
## Model Info

- Model: Random Forest Classifier  
- Accuracy: 98.02%  
- F1-Score: 0.98  
- AUC-ROC: 1.00  
- Tuned using basic hyperparameter optimization

All training steps are documented in the Jupyter notebook.

## Setup & Run

```bash
git clone <repo-link>
cd bank-prediction-app
pip install -r requirements.txt
streamlit run app.py
