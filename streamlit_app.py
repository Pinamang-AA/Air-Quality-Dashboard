import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_csv("C:/Users/pc/Desktop/PROJECT/city_day_cleaned.csv")
    return data

st.title('Air Quality Dashboard')

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Load data
data = load_data(uploaded_file)

# Sidebar filters
st.sidebar.header('Filter Data')
Location = st.sidebar.multiselect('Select Location', data['Location'].unique())
if Location:
    data = data[data['Location'].isin(Location)]

st.header('Raw Data')
st.write(data)

# Data Visualization
st.header('Data Visualization')
fig, ax = plt.subplots()
data['PM2.5'].hist(ax=ax, bins=30)
ax.set_title('PM2.5 Levels')
ax.set_xlabel('PM2.5')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Time Series Analysis
st.header('Time Series Analysis')
fig, ax = plt.subplots()
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
ax.plot(data['Date'], data['PM2.5'], label='PM2.5')
ax.plot(data['Date'], data['PM10'], label='PM10')
ax.set_title('PM2.5 and PM10 Levels Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Levels')
ax.legend()
st.pyplot(fig)

# Correlation Matrix
st.header('Correlation Matrix')
numeric_data = data.select_dtypes(include=[np.number])  # Select only numeric columns
corr = numeric_data.corr()
fig, ax = plt.subplots()
cax = ax.matshow(corr, cmap='coolwarm')
fig.colorbar(cax)
ax.set_xticks(range(len(corr.columns)))
ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)
ax.set_yticklabels(corr.columns)
st.pyplot(fig)

# AQI Prediction
st.header('AQI Prediction')

# Prepare the data
X = data[['PM2.5', 'PM10']]
y = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.write('Mean Absolute Error (MAE):', mae)
st.write('Mean Squared Error (MSE):', mse)
st.write('Root Mean Squared Error (RMSE):', rmse)
st.write('R-squared (R2):', r2)

# Feature Importance
st.header('Feature Importance')
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

fig, ax = plt.subplots()
ax.barh(range(len(indices)), importances[indices], align='center')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([X.columns[i] for i in indices])
ax.set_xlabel('Feature Importance')
st.pyplot(fig)

# Predict AQI based on user input
st.header('Predict AQI')
input_data = {}
for feature in X.columns:
    input_data[feature] = st.number_input(f'Enter {feature}', value=float(data[feature].mean()))
input_df = pd.DataFrame(input_data, index=[0])
prediction = model.predict(input_df)[0]

# Display AQI Meter
st.header('AQI Meter')
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    title={'text': "AQI"},
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
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': prediction}
    }
))
st.plotly_chart(fig)

# AQI Range Information
st.header('AQI Ranges and Meanings')
st.markdown("""
- **0-50 (Green):** Good
- **51-100 (Yellow):** Moderate
- **101-150 (Orange):** Unhealthy for Sensitive Groups
- **151-200 (Red):** Unhealthy
- **201-300 (Purple):** Very Unhealthy
- **301-500 (Maroon):** Hazardous
""")

# Data Summary
st.header('Data Summary')
st.write(data.describe())

# Download Data
st.sidebar.header('Download Data')
if st.sidebar.button('Download CSV'):
    filtered_data = data.to_csv().encode('utf-8')
    st.sidebar.download_button('Download filtered data', data=filtered_data, file_name='filtered_air_quality_data.csv', mime='text/csv')
