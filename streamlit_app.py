import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# Load Data Function
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("city_day_cleaned.csv")
    return data

# Set title and sidebar
st.title('📊 Air Quality Dashboard')

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data
data = load_data(uploaded_file)

# Sidebar filters
st.sidebar.header('📍 Filters')
Location = st.sidebar.multiselect('Select Location', data['Location'].unique())
if Location:
    data = data[data['Location'].isin(Location)]

# Layout grid for visualizations
col1, col2 = st.columns(2)

# Column 1: Raw Data and PM2.5 Histogram
with col1:
    st.header('📝 Raw Data')
    st.write(data)

    st.header('📊 PM2.5 Distribution')
    fig, ax = plt.subplots()
    data['PM2.5'].hist(ax=ax, bins=30, color='skyblue')
    ax.set_title('PM2.5 Levels', fontsize=16)
    ax.set_xlabel('PM2.5 (µg/m³)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    st.pyplot(fig)

# Column 2: Time Series Analysis and AQI Meter
with col2:
    st.header('📊 Monthly Trends')
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')
    numeric_columns = ['PM2.5', 'PM10']
    data_numeric = data[['Date'] + numeric_columns]
    monthly_data = data_numeric.set_index('Date').resample('M').mean()
    
    fig, ax = plt.subplots()
    ax.plot(monthly_data.index, monthly_data['PM2.5'], label='PM2.5')
    ax.plot(monthly_data.index, monthly_data['PM10'], label='PM10')
    ax.set_title('Monthly Average PM2.5 and PM10 Levels Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Levels')
    ax.legend()
    st.pyplot(fig)

    # AQI Prediction based on user input
    st.header('🌡️ AQI Meter')
    
    # Feature Inputs from user
    input_data = {}
    for feature in ['PM2.5', 'PM10']:
        input_data[feature] = st.number_input(f'Enter {feature}', value=float(data[feature].mean()))
    
    input_df = pd.DataFrame(input_data, index=[0])
    
    # Check for AQI column existence and handle missing values
    if 'AQI' in data.columns and not data['AQI'].isnull().values.any():
        # Prepare the data for model training
        X = data[['PM2.5', 'PM10']]
        y = data['AQI']

        # Split data and train the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediction based on user input
        prediction = model.predict(input_df)[0]

        # Display AQI Meter
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={'text': "Predicted AQI"},
            gauge={
                'axis': {'range': [0, 500]},
                'steps': [
                    {'range': [0, 50], 'color': "green"},
                    {'range': [51, 100], 'color': "yellow"},
                    {'range': [101, 150], 'color': "orange"},
                    {'range': [151, 200], 'color': "red"},
                    {'range': [201, 300], 'color': "purple"},
                    {'range': [301, 500], 'color': "maroon"},
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': prediction}
            }
        ))
        st.plotly_chart(fig)
    else:
        st.warning('AQI data is missing or incomplete. Please upload data with AQI values.')

# Next row: Correlation matrix and model evaluation
col3, col4 = st.columns(2)

# Column 3: Correlation Matrix
with col3:
    st.header('📊 Correlation Matrix')
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots()
    cax = ax.matshow(corr, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

# Column 4: AQI Prediction Results
with col4:
    st.header('🤖 AQI Prediction Evaluation')
    
    if 'AQI' in data.columns and not data['AQI'].isnull().values.any():
        # Model already trained, evaluate the performance
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        st.write(f'**Mean Absolute Error (MAE):** {mae:.2f}')
        st.write(f'**Mean Squared Error (MSE):** {mse:.2f}')
        st.write(f'**Root Mean Squared Error (RMSE):** {rmse:.2f}')
        st.write(f'**R-squared (R2):** {r2:.2f}')
    else:
        st.warning('AQI data is missing or incomplete for model evaluation.')

# AQI Range Information
st.header('📘 AQI Ranges and Meanings')
st.markdown("""
- **0-50 (Green):** Good
- **51-100 (Yellow):** Moderate
- **101-150 (Orange):** Unhealthy for Sensitive Groups
- **151-200 (Red):** Unhealthy
- **201-300 (Purple):** Very Unhealthy
- **301-500 (Maroon):** Hazardous
""")

# Data Summary
st.header('📊 Data Summary')
st.write(data.describe())

# Download Data Button
st.sidebar.header('📥 Download Data')
if st.sidebar.button('Download CSV'):
    filtered_data = data.to_csv().encode('utf-8')
    st.sidebar.download_button('Download filtered data', data=filtered_data, file_name='filtered_air_quality_data.csv', mime='text/csv')
