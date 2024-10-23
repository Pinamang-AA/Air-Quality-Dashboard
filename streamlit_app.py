import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load Data Function
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("city_day_cleaned.csv")
    return data

# Set title and page layout
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title('📊 Modern Air Quality Dashboard')

# Sidebar for file upload and filters
st.sidebar.header("📂 Data Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Load data
data = load_data(uploaded_file)

# Sidebar filters
Location = st.sidebar.multiselect('Select Location(s)', data['Location'].unique(), default=data['Location'].unique())
if Location:
    data = data[data['Location'].isin(Location)]

# Ensure the Date column is properly parsed
try:
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert to datetime, coerce errors
    data = data.dropna(subset=['Date'])  # Drop rows where 'Date' could not be parsed
except Exception as e:
    st.error(f"Error parsing date: {e}")

# Dashboard layout: 3 sections in a grid
st.markdown("## Dashboard Overview")

# 1st Row: Raw Data and Distribution Plots
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📝 Raw Data")
    st.write(data)

    st.markdown("### 📊 PM2.5 & PM10 Distribution")
    # Plotly Histogram for Distribution
    fig = px.histogram(data, x='PM2.5', nbins=30, title="PM2.5 Distribution", color_discrete_sequence=['#1f77b4'])
    fig.update_layout(xaxis_title='PM2.5 (µg/m³)', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Plotly Histogram for PM10
    fig = px.histogram(data, x='PM10', nbins=30, title="PM10 Distribution", color_discrete_sequence=['#ff7f0e'])
    fig.update_layout(xaxis_title='PM10 (µg/m³)', yaxis_title='Frequency')
    st.plotly_chart(fig, use_container_width=True)

# Time Series Analysis: Monthly Trends (2nd row)
st.markdown("## Monthly Trends")

# Ensure only numeric columns are resampled
numeric_columns = data.select_dtypes(include=np.number).columns

if 'Date' in data.columns:
    monthly_data = data.set_index('Date').resample('M')[numeric_columns].mean().reset_index()

    # Create a dual-line plot with Plotly for Monthly Trends
    fig = px.line(monthly_data, x='Date', y=['PM2.5', 'PM10'], title="Monthly Average PM2.5 and PM10 Levels",
                  labels={'value': 'Levels (µg/m³)', 'variable': 'Pollutant'}, color_discrete_sequence=['#1f77b4', '#ff7f0e'])
    fig.update_layout(xaxis_title='Date', yaxis_title='Pollutant Level')
    st.plotly_chart(fig, use_container_width=True)

# AQI Prediction Section: 3rd row
st.markdown("## 🤖 Predict AQI Based on PM2.5 and PM10 Inputs")

# Collect user inputs for AQI prediction
col3, col4, col5 = st.columns(3)
with col3:
    user_pm25 = st.number_input("Enter PM2.5 value", value=float(data['PM2.5'].mean()))
with col4:
    user_pm10 = st.number_input("Enter PM10 value", value=float(data['PM10'].mean()))

# Prepare model data
if 'AQI' in data.columns and not data['AQI'].isnull().values.any():
    X = data[['PM2.5', 'PM10']]
    y = data['AQI']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make prediction with user input
    input_data = pd.DataFrame({'PM2.5': [user_pm25], 'PM10': [user_pm10]})
    predicted_aqi = model.predict(input_data)[0]

    with col5:
        st.metric(label="Predicted AQI", value=f"{predicted_aqi:.2f}")

    # Display AQI Gauge
    st.markdown("### 🌡️ AQI Prediction Meter")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_aqi,
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
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Correlation Matrix and Model Evaluation (4th row)
st.markdown("## 📊 Model Evaluation and Correlation")

col6, col7 = st.columns(2)

with col6:
    st.markdown("### 🔑 Feature Importance (PM2.5 & PM10)")
    importances = model.feature_importances_
    fig = px.bar(x=X.columns, y=importances, title="Feature Importance", labels={'x': 'Feature', 'y': 'Importance'})
    st.plotly_chart(fig, use_container_width=True)

with col7:
    st.markdown("### 📊 Correlation Matrix")
    corr = data[['PM2.5', 'PM10', 'AQI']].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
    st.plotly_chart(fig, use_container_width=True)

# AQI Ranges Information
st.markdown("## 📘 AQI Ranges and Meanings")
st.markdown("""
- **0-50 (Green):** Good  
- **51-100 (Yellow):** Moderate  
- **101-150 (Orange):** Unhealthy for Sensitive Groups  
- **151-200 (Red):** Unhealthy  
- **201-300 (Purple):** Very Unhealthy  
- **301-500 (Maroon):** Hazardous  
""")

# Download Data Button
st.sidebar.header('📥 Download Data')
if st.sidebar.button('Download CSV'):
    filtered_data = data.to_csv().encode('utf-8')
    st.sidebar.download_button('Download filtered data', data=filtered_data, file_name='filtered_air_quality_data.csv', mime='text/csv')
