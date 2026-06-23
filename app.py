import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

# 1. ऐप का टाइटल और डिज़ाइन
st.set_page_config(page_title="Coffee Sales Forecast", layout="wide")
st.title("☕ Coffee Sales Analytics & Future Prediction")
st.write("Jupyter में तैयार किया गया डेटा अब Streamlit पर लाइव है!")

# अपनी फाइल का नाम यहाँ सही से लिखें
csv_filename = "Coffee_Sales.csv" 

# 2. डेटा लोड करने का फ़ंक्शन
@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename) 
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Hour'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p', errors='coerce').dt.hour
    return df

# बिना try-except के सीधा फाइल चेक करना (इससे कोई इंडेंटेशन एरर नहीं आएगी)
if not os.path.exists(csv_filename):
    st.error(f"⚠️ फ़ाइल '{csv_filename}' नहीं मिली! कृपया लाइन नंबर 14 पर अपनी सही CSV फ़ाइल का नाम लिखें।")
else:
    df = load_data("Coffee_Sales.csv")
    
    # --- UI लेआउट: मुख्य आँकड़े (KPI Cards) ---
    col1, col2, col3 = st.columns(3)
    col1.metric("कुल कमाई (Total Revenue)", f"₹{df['Money'].sum():,}")
    col2.metric("कुल ऑर्डर्स (Total Orders)", f"{len(df)}")
    col3.metric("औसत Order वैल्यू (Avg Ticket)", f"₹{df['Money'].mean():.2f}")

    # --- साइडबार में यूज़र इनपुट ---
    st.sidebar.header("Prediction Settings")
    days_to_predict = st.sidebar.slider("कितने दिनों की सेल्स प्रेडिक्ट करनी है?", min_value=1, max_value=30, value=7)

    # --- डेटा को ML के लिए तैयार करना ---
    daily_sales = df.groupby('Date')['Money'].sum().reset_index()
    daily_sales = daily_sales.sort_values('Date').reset_index(drop=True)
    daily_sales['Day_Num'] = daily_sales.index

    X = daily_sales[['Day_Num']]
    y = daily_sales['Money']

    # मॉडल ट्रेनिंग
    model = LinearRegression()
    model.fit(X, y)

    # --- प्रेडिक्शन बटन ---
    if st.sidebar.button("Predict Future Sales 🚀"):
        st.subheader(f"🔮 अगले {days_to_predict} दिनों की सेल्स का अनुमान")
        
        # भविष्य के दिनों की गणना
        last_day = daily_sales['Day_Num'].max()
        future_days = np.array([[last_day + i] for i in range(1, days_to_predict + 1)])
        future_preds = model.predict(future_days)
        
        # भविष्य की तारीखें बनाना
        last_date = daily_sales['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
        
        # प्रेडिक्शन डेटाफ़्रेम
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Sales': future_preds
        })
        
        st.write("Forecasted DataFrame:")
        st.dataframe(forecast_df.style.format({'Predicted_Sales': '₹{:.2f}'}))
        
        # --- विज़ुअलाइज़ेशन (लाइव ग्राफ़) ---
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_sales['Date'], daily_sales['Money'], label='Past Sales', color='blue', marker='o')
        ax.plot(forecast_df['Date'], forecast_df['Predicted_Sales'], label='Forecast', color='red', linestyle='--', marker='s')
        
        ax.set_title("Coffee Sales Trend & Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales (₹)")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
