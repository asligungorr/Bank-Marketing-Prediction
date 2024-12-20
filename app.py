import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Model ve scaler'ları yükleme
@st.cache_resource
def load_model():
    try:
        model = joblib.load('BSA_model.pkl')
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None

@st.cache_resource
def load_encoders_and_scaler():
    try:
        # Encoder ve scaler'ları yükle
        encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('standard_scaler.pkl')
        return encoders, scaler
    except Exception as e:
        st.error(f"Encoder ve scaler yüklenirken hata oluştu: {e}")
        return None, None

def prepare_input_data(input_data):
    # Encoder ve scaler'ı yükle
    encoders, scaler = load_encoders_and_scaler()
    if encoders is None or scaler is None:
        raise Exception("Encoder veya scaler yüklenemedi")
    
    # Kategorik değişkenleri dönüştür
    categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 
                       'contact', 'month', 'day_of_week', 'poutcome']
    
    data_processed = input_data.copy()
    
    for col in categorical_cols:
        if col in encoders:
            data_processed[col] = encoders[col].transform(input_data[col])
    
    # Sayısal değişkenleri ölçeklendir
    numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    data_processed[numerical_cols] = scaler.transform(input_data[numerical_cols])
    
    return data_processed

def show_model_details():
    st.title('Model Details')
    
    # Model Performance Metrics
    st.header('Model Performance')
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Model Accuracy", value="98.02%")
    with col2:
        st.metric(label="F1 Score", value="0.98")
    with col3:
        st.metric(label="AUC-ROC", value="0.98")
    
    # Model Information
    st.header('Model Information')
    st.write("""
    - **Selected Model**: Random Forest Classifier
    - **Hyperparameters**:
        - max_depth: 20
        - min_samples_leaf: 1
        - min_samples_split: 2
        - n_estimators: 200
    """)
    
    # Confusion Matrix
    st.header('Confusion Matrix')
    confusion_matrix = [
        [705, 29],
        [0, 734]
    ]
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=['Predicted No', 'Predicted Yes'],
        y=['Actual No', 'Actual Yes'],
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        colorscale='Viridis'
    ))
    
    fig_cm.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label'
    )
    
    st.plotly_chart(fig_cm)
    
    # Model Comparison
    st.header('Model Comparison')
    model_comparison = {
        'Model': ['Logistic Regression', 'Random Forest', 'Neural Network', 'Gradient Boosting'],
        'Accuracy': [71.59, 98.02, 93.73, 87.39],
        'F1 Score': [0.71, 0.98, 0.93, 0.87]
    }
    
    df_comparison = pd.DataFrame(model_comparison)
    fig_comparison = px.bar(df_comparison, x='Model', y=['Accuracy', 'F1 Score'],
                          barmode='group', title='Model Performance Comparison')
    st.plotly_chart(fig_comparison)
    
    # Feature Importance
    st.header('Feature Importance')
    feature_importance = {
        'Feature': ['euribor3m', 'nr.employed', 'emp.var.rate', 'cons.price.idx', 'pdays'],
        'Importance': [0.25, 0.20, 0.15, 0.12, 0.10]
    }
    
    df_importance = pd.DataFrame(feature_importance)
    fig_importance = px.bar(df_importance, x='Feature', y='Importance',
                          title='Top 5 Most Important Features')
    st.plotly_chart(fig_importance)

def main():
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Prediction', 'Model Details'])
    
    if page == 'Prediction':
        st.title('Bank Marketing Prediction App')
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Input fields
        st.subheader('Müşteri Bilgileri')
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Yaş', min_value=18, max_value=100, value=30)
            job = st.selectbox('Meslek', options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                                'management', 'retired', 'self-employed', 'services', 
                                                'student', 'technician', 'unemployed', 'unknown'])
            marital = st.selectbox('Medeni Durum', options=['married', 'divorced', 'single'])
            education = st.selectbox('Eğitim', options=['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                      'illiterate', 'professional.course', 'university.degree', 
                                                      'unknown'])
            default = st.selectbox('Kredi Temerrüdü', options=['no', 'yes'])
            housing = st.selectbox('Konut Kredisi', options=['no', 'yes'])
            loan = st.selectbox('Kişisel Kredi', options=['no', 'yes'])
            
        with col2:
            contact = st.selectbox('İletişim Türü', options=['cellular', 'telephone', 'unknown'])
            month = st.selectbox('Ay', options=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                              'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            day_of_week = st.selectbox('Haftanın Günü', options=['mon', 'tue', 'wed', 'thu', 'fri'])
            campaign = st.number_input('Kampanya İletişim Sayısı', min_value=1, max_value=50, value=1)
            pdays = st.number_input('Son İletişimden Bu Yana Geçen Gün', min_value=0, value=0)
            previous = st.number_input('Önceki Kampanya İletişim Sayısı', min_value=0, value=0)
            poutcome = st.selectbox('Önceki Kampanya Sonucu', options=['failure', 'nonexistent', 'success', 'unknown'])
        
        st.subheader('Ekonomik Göstergeler')
        col3, col4 = st.columns(2)
        
        with col3:
            emp_var_rate = st.number_input('İstihdam Değişim Oranı', value=0.0, step=0.1)
            cons_price_idx = st.number_input('Tüketici Fiyat Endeksi', value=93.2, step=0.1)
            cons_conf_idx = st.number_input('Tüketici Güven Endeksi', value=-36.4, step=0.1)
        
        with col4:
            euribor3m = st.number_input('Euribor 3 Aylık Oran', value=4.857, step=0.001)
            nr_employed = st.number_input('Çalışan Sayısı', value=5191.0, step=1.0)

        if st.button('Tahmin Et'):
            try:
                # Veriyi DataFrame'e dönüştür
                input_data = pd.DataFrame({
                    'age': [age],
                    'job': [job],
                    'marital': [marital],
                    'education': [education],
                    'default': [default],
                    'housing': [housing],
                    'loan': [loan],
                    'contact': [contact],
                    'month': [month],
                    'day_of_week': [day_of_week],
                    'campaign': [campaign],
                    'pdays': [pdays],
                    'previous': [previous],
                    'poutcome': [poutcome],
                    'emp.var.rate': [emp_var_rate],
                    'cons.price.idx': [cons_price_idx],
                    'cons.conf.idx': [cons_conf_idx],
                    'euribor3m': [euribor3m],
                    'nr.employed': [nr_employed]
                })
                
                # Veri ön işleme
                processed_data = prepare_input_data(input_data)
                
                # Tahmin
                prediction = model.predict(processed_data)
                probability = model.predict_proba(processed_data)
                
                # Sonuçları göster
                if prediction[0] == 1:
                    st.success('Tahmin: Müşteri vadeli mevduat ürününü satın alma olasılığı YÜKSEK')
                else:
                    st.error('Tahmin: Müşteri vadeli mevduat ürününü satın alma olasılığı DÜŞÜK')
                    
                st.write(f'Satın alma olasılığı: {probability[0][1]:.2%}')
                
            except Exception as e:
                st.error(f"Tahmin yapılırken bir hata oluştu: {e}")
    
    else:  # Model Details page
        show_model_details()

if __name__ == '__main__':
    main()
