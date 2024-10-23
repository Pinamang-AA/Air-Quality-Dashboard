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

# Convert 'Date' column to datetime if not already in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sidebar filters
st.sidebar.header('📍 Filters')
st.sidebar.write("Select a location to analyze its air quality data.")
Location = st.sidebar.multiselect('Select Location', data['Location'].unique())
if Location:
    data = data[data['Location'].isin(Location)]

# Date range filter
st.sidebar.write("Select Date Range")
start_date = st.sidebar.date_input("Start date", value=data['Date'].min().date())
end_date = st.sidebar.date_input("End date", value=data['Date'].max().date())

# Ensure the selected range is valid
if start_date > end_date:
    st.sidebar.error("Error: End date must fall after the start date.")

# Filter the data based on the selected date range
data = data[(data['Date'] >= pd.to_datetime(start_date)) & (data['Date'] <= pd.to_datetime(end_date))]

# Display Raw Data
st.header('📝 Raw Data')
st.write(data)

# AQI Prediction based on user input
st.header('📈 Predict AQI')
input_data = {}
pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']

for pollutant in pollutants:
    input_data[pollutant] = st.number_input(f'Enter {pollutant}', value=float(data[pollutant].mean()))
    
input_df = pd.DataFrame(input_data, index=[0])
model = RandomForestRegressor(n_estimators=100, random_state=42)
X = data[pollutants]
y = data['AQI']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
prediction = model.predict(input_df)[0]

# Display AQI Meter
st.header('🌡️ AQI Meter')
aqi_status = "Good" if prediction <= 50 else "Moderate" if prediction <= 100 else "Unhealthy"
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': f"Predicted AQI: {aqi_status}"},
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

# Real-time Data Visualization (Past 48 hours)
st.header('⏳ Real-time Air Quality Data (Past 48 hours)')
last_48h = data.sort_values(by='Date').tail(48)  # Adjust 'tail(48)' based on your data frequency
cols = pollutants + ['Temp', 'Humidity', 'Pressure']

# Bar plots for pollutants over the last 48 hours
for col in pollutants:
    fig, ax = plt.subplots()
    ax.bar(last_48h['Date'], last_48h[col], color='skyblue')
    ax.set_title(f'{col} Levels - Past 48 hours', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'{col} (µg/m³)', fontsize=12)
    st.pyplot(fig)

# Display Weather Information
st.header('🌦️ Weather Information')
st.metric(label="Temperature (°C)", value=last_48h['Temp'].mean())
st.metric(label="Humidity (%)", value=last_48h['Humidity'].mean())
st.metric(label="Pressure (hPa)", value=last_48h['Pressure'].mean())

# Correlation Matrix
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
