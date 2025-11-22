import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import ta

st.set_page_config(page_title="AI Market Analyzer", layout="wide")

st.title("📊 AI Market Analyzer — Web Version")
st.write("تحليل السوق باستخدام الذكاء الاصطناعي + MACD + RSI + Bollinger")

# --- INPUTS ---
symbol = st.text_input("Symbol", "AAPL")
period = st.selectbox("Period", ["7d","1mo","3mo","6mo","1y"])
interval = st.selectbox("Interval", ["1m","5m","15m","1h","4h","1d"])

if st.button("Run Analysis"):
    
    st.info("⏳ Fetching data...")

    df = yf.download(
        tickers=symbol,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if df.empty:
        st.error("No data found!")
        st.stop()

    # --- Fix MultiIndex ---
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
    df["Macd"] = macd.macd().astype(float)
    df["Signal"] = macd.macd_signal().astype(float)
    boll = ta.volatility.BollingerBands(close_series)
    df["Boll_h"] = boll.bollinger_hband().astype(float)
    df["Boll_l"] = boll.bollinger_lband().astype(float)

    df.dropna(inplace=True)

    # AI Prediction
    df["Prediction"] = df["Close"].shift(-1)
    df_model = df.dropna()

    features = ["Close","Rsi","Macd","Signal","Boll_h","Boll_l"]
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

    trend = "⬆️ UP (صعود)" if pred > current_price else "⬇️ DOWN (هبوط)"

    # Buy/Sell Signals
    buy = (df["Macd"] > df["Signal"]) & (df["Macd"].shift(1) <= df["Signal"].shift(1))
    sell = (df["Macd"] < df["Signal"]) & (df["Macd"].shift(1) >= df["Signal"].shift(1))

    df["Buy"] = np.where(buy, df["Boll_l"], np.nan)
    df["Sell"] = np.where(sell, df["Boll_h"], np.nan)

    # ---- Display Result ----
    st.subheader("📘 Analysis Result")
    st.write(f"**Current Price:** {current_price}")
    st.write(f"**Predicted Price:** {pred:.2f}")
    st.write(f"**Trend:** {trend}")

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df.index, df["Close"], label="Price", color="cyan")
    ax.scatter(df.index, df["Buy"], color="lime", marker="^", s=120, label="Buy")
    ax.scatter(df.index, df["Sell"], color="red", marker="v", s=120, label="Sell")
    ax.scatter(df.index[-1], pred, color="yellow", marker="*", s=200, label="Prediction")

    ax.set_title(f"{symbol.upper()} Price Chart")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    st.success("✔️ Analysis Completed!")