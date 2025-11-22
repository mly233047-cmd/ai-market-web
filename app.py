import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import ta
import streamlit as st
import time

st.set_page_config(page_title="AI Market Analyzer", layout="wide")

st.title("📊 AI Market Analyzer — Web Version")

# --- Sidebar ---
symbol = st.sidebar.text_input("Symbol", "AAPL")
period = st.sidebar.selectbox("Period", ["7d","1mo","3mo","6mo","1y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"], index=2)

auto_refresh = st.sidebar.checkbox("Auto Refresh (كل 60 ثانية)")
run = st.sidebar.button("Run Analysis")

if auto_refresh:
    st.experimental_rerun()

if run or auto_refresh:
    st.info("Fetching data...")
    
    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        st.error("No data found for this symbol.")
        st.stop()

    # Fix MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Date" in df.columns:
        df.set_index("Date", inplace=True)
    elif "Datetime" in df.columns:
        df.set_index("Datetime", inplace=True)

    df.columns = [c.capitalize() for c in df.columns]

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    close = df["Close"].values.flatten()
    close_series = pd.Series(close, index=df.index)

    # Indicators
    df["Rsi"] = ta.momentum.RSIIndicator(close_series).rsi()
    macd = ta.trend.MACD(close_series)
    df["Macd"] = macd.macd()
    df["Signal"] = macd.macd_signal()

    boll = ta.volatility.BollingerBands(close_series)
    df["Boll_high"] = boll.bollinger_hband()
    df["Boll_low"] = boll.bollinger_lband()

    df.dropna(inplace=True)

    # AI Prediction
    df["Prediction"] = df["Close"].shift(-1)
    df_model = df.dropna()

    features = ["Close","Rsi","Macd","Signal","Boll_high","Boll_low"]
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

    pred = float(model.predict(X.iloc[-1].values.reshape(1,-1))[0])
    price = float(df["Close"].iloc[-1])

    trend = "🔺 UP (صعود)" if pred > price else "🔻 DOWN (هبوط)"

    # Buy/Sell Signals
    buy = (df["Macd"] > df["Signal"]) & (df["Macd"].shift(1) <= df["Signal"].shift(1))
    sell = (df["Macd"] < df["Signal"]) & (df["Macd"].shift(1) >= df["Signal"].shift(1))

    df["Buy_sig"] = np.where(buy, df["Boll_low"], np.nan)
    df["Sell_sig"] = np.where(sell, df["Boll_high"], np.nan)

    # --- Display Results ---
    st.subheader("🟦 Analysis Result")
    st.write(f"**Current Price:** {price:.2f}")
    st.write(f"**Predicted Next Price:** {pred:.2f}")
    st.write(f"**Trend:** {trend}")

    # --- Chart ---
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["Close"], label="Price", color="cyan")
    ax.scatter(df.index, df["Buy_sig"], color="lime", marker="^", s=150, label="Buy")
    ax.scatter(df.index, df["Sell_sig"], color="red", marker="v", s=150, label="Sell")

    # Prediction star
    ax.scatter(df.index[-1], pred, color="yellow", s=250, marker="*", label="Prediction")

    ax.set_title(f"{symbol.upper()} Price Chart")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    if auto_refresh:
        time.sleep(60)
        st.experimental_rerun()