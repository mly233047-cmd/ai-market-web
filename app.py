import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import ta
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Market Analyzer", layout="wide")

st.title("📊 AI Market Analyzer — Web Version")

# --- Inputs ---
symbol = st.text_input("Enter symbol (e.g., GC=F, EURUSD=X, BTC-USD):", "AAPL")
period = st.selectbox("Select period:", ["1mo", "3mo", "6mo", "1y"], index=1)
interval = st.selectbox("Select interval:", ["1m", "5m", "15m", "1h", "4h", "1d"], index=3)

if st.button("Run Analysis"):
    st.write("⏳ Fetching data...")

    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        st.error("❌ No data found. Try another symbol.")
        st.stop()

    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    close_series = pd.Series(df["Close"].values, index=df.index)

    # Indicators
    df["RSI"] = ta.momentum.RSIIndicator(close_series).rsi()
    macd = ta.trend.MACD(close_series)
    df["MACD"] = macd.macd()
    df["Signal"] = macd.macd_signal()
    boll = ta.volatility.BollingerBands(close_series)
    df["Bollinger_High"] = boll.bollinger_hband()
    df["Bollinger_Low"] = boll.bollinger_lband()

    df.dropna(inplace=True)

    # Prepare data for AI model
    df["Prediction"] = df["Close"].shift(-1)
    df_model = df.dropna()

    features = ["Close", "RSI", "MACD", "Signal", "Bollinger_High", "Bollinger_Low"]
    X = df_model[features]
    y = df_model["Prediction"]

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror"
    )
    model.fit(X, y)

    last_row = X.iloc[-1].values.reshape(1, -1)
    pred = float(model.predict(last_row)[0])
    current_price = float(df["Close"].iloc[-1])

    trend = "🔺 UP (صعود)" if pred > current_price else "🔻 DOWN (هبوط)"

    # Display results
    st.subheader("📘 Analysis Result")
    st.write(f"**Current Price:** {current_price:.2f}")
    st.write(f"**Predicted Price:** {pred:.2f}")
    st.write(f"**Trend:** {trend}")

    # Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Price", color="cyan")
    ax.scatter(df.index[-1], pred, color="yellow", s=200, marker="*", label="Prediction")
    ax.set_title(f"{symbol.upper()} Price Chart")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)