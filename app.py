import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Load model and scalers
@st.cache_resource
def load_model():
    try:
        model = joblib.load('BSA_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def prepare_input_data(df):
    """Performs data preprocessing steps"""
    # Mapping for categorical variables
    job_mapping = {
        'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
        'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
        'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11
    }
    
    marital_mapping = {'divorced': 0, 'married': 1, 'single': 2}
    
    education_mapping = {
        'basic.4y': 0, 'basic.6y': 1, 'basic.9y': 2, 'high.school': 3,
        'illiterate': 4, 'professional.course': 5, 'university.degree': 6, 'unknown': 7
    }
    
    default_mapping = {'no': 0, 'yes': 1}
    housing_mapping = {'no': 0, 'yes': 1}
    loan_mapping = {'no': 0, 'yes': 1}
    contact_mapping = {'cellular': 0, 'telephone': 1, 'unknown': 2}
    
    month_mapping = {
        'jan': 0, 'feb': 1, 'mar': 2, 'apr': 3, 'may': 4, 'jun': 5,
        'jul': 6, 'aug': 7, 'sep': 8, 'oct': 9, 'nov': 10, 'dec': 11
    }
    
    day_mapping = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4}
    poutcome_mapping = {'failure': 0, 'nonexistent': 1, 'success': 2, 'unknown': 3}

    # Convert categorical variables
    df['job'] = df['job'].map(job_mapping)
    df['marital'] = df['marital'].map(marital_mapping)
    df['education'] = df['education'].map(education_mapping)
    df['default'] = df['default'].map(default_mapping)
    df['housing'] = df['housing'].map(housing_mapping)
    df['loan'] = df['loan'].map(loan_mapping)
    df['contact'] = df['contact'].map(contact_mapping)
    df['month'] = df['month'].map(month_mapping)
    df['day_of_week'] = df['day_of_week'].map(day_mapping)
    df['poutcome'] = df['poutcome'].map(poutcome_mapping)

    # Standardize numerical variables
    scaler = StandardScaler()
    numerical_cols = ['age', 'campaign', 'pdays', 'previous', 'emp.var.rate',
                     'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    
    # Normalize numerical variables
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def show_data_details():
    st.title('Data Details and Analysis')
    
    st.write("The bank marketing dataset data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).")
    st.write("There is no null value.")
    st.write("Link: https://archive.ics.uci.edu/ml/datasets/bank+marketing")

    pie_chart = Image.open('pie_chart.png')
    st.header('Pie Chart of dataset distribution')
    st.write("The number of 'yes' asweres was 3368 and the number of 'no' values was 451. This means that dataset was unbalanced.")
    st.image(pie_chart, use_container_width=True)


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
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label'
    )
    
    st.plotly_chart(fig_cm)

    report_table = Image.open('report_table.jpg')
    roc_curve = Image.open('roc_curve.png')
    st.header('Analysis of Model')
    st.image(report_table, use_container_width=True)
    st.header('ROC Curve')
    st.image(roc_curve, use_container_width=True)

def main():
    st.sidebar.title('Sidebar')
    st.sidebar.markdown("""
    **Choose your option below:**
    """)
    
    # Create buttons for navigation
    prediction_button = st.sidebar.button('Prediction')
    model_details_button = st.sidebar.button('Model Details')
    data_details_button = st.sidebar.button('Data Details')

    if prediction_button:
        page = 'Prediction'
    elif model_details_button:
        page = 'Model Details'
    elif data_details_button:
        page = 'Data Details'
    else:
        page = 'Prediction'  # Default page

    if page == 'Prediction':
        st.title('Bank Marketing Prediction App')
        
        # Load model
        model = load_model()
        if model is None:
            return
        
        # Input fields
        st.subheader('Customer Information')
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input('Age', min_value=18, max_value=100, value=30)
            job = st.selectbox('Job', options=['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 
                                                'management', 'retired', 'self-employed', 'services', 
                                                'student', 'technician', 'unemployed', 'unknown'])
            marital = st.selectbox('Marital Status', options=['married', 'divorced', 'single'])
            education = st.selectbox('Education', options=['basic.4y', 'basic.6y', 'basic.9y', 'high.school',
                                                      'illiterate', 'professional.course', 'university.degree', 
                                                      'unknown'])
            default = st.selectbox('Credit Default', options=['no', 'yes'])
            housing = st.selectbox('Housing Loan', options=['no', 'yes'])
            loan = st.selectbox('Personal Loan', options=['no', 'yes'])
            
        with col2:
            contact = st.selectbox('Contact Type', options=['cellular', 'telephone', 'unknown'])
            month = st.selectbox('Month', options=['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
            day_of_week = st.selectbox('Day of the Week', options=['mon', 'tue', 'wed', 'thu', 'fri'])
            campaign = st.number_input('Number of Campaign Contacts', min_value=1, max_value=50, value=1)
            pdays = st.number_input('Days Since Last Contact', min_value=0, value=0)
            previous = st.number_input('Previous Campaign Contacts', min_value=0, value=0)
            poutcome = st.selectbox('Previous Campaign Outcome', options=['failure', 'nonexistent', 'success', 'unknown'])
        
        st.subheader('Economic Indicators')
        col3, col4 = st.columns(2)
        
        with col3:
            emp_var_rate = st.number_input('Employment Variation Rate', value=0.0, step=0.1)
            cons_price_idx = st.number_input('Consumer Price Index', value=93.2, step=0.1)
            cons_conf_idx = st.number_input('Consumer Confidence Index', value=-36.4, step=0.1)
        
        with col4:
            euribor3m = st.number_input('Euribor 3 Month Rate', value=4.857, step=0.001)
            nr_employed = st.number_input('Number of Employees', value=5191.0, step=1.0)

        if st.button('Predict'):
            try:
                # Convert data to DataFrame
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
                
                # Preprocess data
                input_data = prepare_input_data(input_data)
                
                # Make prediction
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)
                
                if prediction == 1:
                    st.warning('Prediction: The customer is likely to purchase the term deposit product.')
                else:
                    st.warning('Prediction: The customer is unlikely to purchase the term deposit product.')
                
                st.write(f"Probability of customer purchasing: {probability[0][1]*100:.2f}%")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    elif page == 'Model Details':
        show_model_details()
    elif page == 'Data Details':
        show_data_details()

if __name__ == "__main__":
    main()
