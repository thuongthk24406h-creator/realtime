import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Real-Time BTC Candlestick & Anomaly",
    page_icon="🕯️",
    layout="wide"
)

st.title("🕯️ Real-Time BTC/USDT Candlestick Chart & Anomaly Detection")
st.caption("Hệ thống phân tích Biểu đồ Nến Nhật (Candlestick) theo thời gian thực kết hợp AI phát hiện biến động giá bất thường")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Cấu hình Nến & Luồng Dữ liệu")
candle_duration = st.sidebar.slider("Khung thời gian 1 cây Nến (Giây):", min_value=5, max_value=60, value=10, step=5)
window_candles = st.sidebar.slider("Số lượng Nến hiển thị (Window Size):", min_value=15, max_value=100, value=30)
contamination = st.sidebar.slider("Tỉ lệ nhạy Anomaly (Contamination):", min_value=0.01, max_value=0.20, value=0.05, step=0.01)

# --- Khởi tạo Session State ---
if "raw_ticks" not in st.session_state:
    st.session_state.raw_ticks = []  # Lưu các tick giá nguyên bản

if "df_candles" not in st.session_state:
    st.session_state.df_candles = pd.DataFrame(columns=["Timestamp", "Open", "High", "Low", "Close", "Is_Anomaly"])

# Hàm lấy giá BTC/USDT trực tiếp từ Binance
def get_binance_price():
    url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
    try:
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            return float(res.json()["price"])
    except:
        pass
    return None

placeholder = st.empty()

# --- Vòng lặp thu thập & tạo nến Real-time ---
while True:
    current_time = pd.Timestamp.now()
    price = get_binance_price()

    if price is not None:
        st.session_state.raw_ticks.append({"time": current_time, "price": price})

    # Gom nhóm dữ liệu tick thành các Cây Nến (OHLC) dựa trên candle_duration
    if len(st.session_state.raw_ticks) > 0:
        df_ticks = pd.DataFrame(st.session_state.raw_ticks)
        
        # Tạo mốc thời gian làm tròn cho khung nến
        df_ticks["Candle_Time"] = df_ticks["time"].dt.floor(f"{candle_duration}s")
        
        # Gom nhóm tính toán OHLC
        candles_list = []
        for candle_time, group in df_ticks.groupby("Candle_Time"):
            open_p = group["price"].iloc[0]
            high_p = group["price"].max()
            low_p = group["price"].min()
            close_p = group["price"].iloc[-1]
            
            candles_list.append({
                "Timestamp": candle_time,
                "Open": open_p,
                "High": high_p,
                "Low": low_p,
                "Close": close_p,
                "Is_Anomaly": 0
            })
            
        st.session_state.df_candles = pd.DataFrame(candles_list)

    # Lấy N nến gần nhất để chạy AI & Vẽ biểu đồ
    df_window = st.session_state.df_candles.tail(window_candles).copy()

    # --- Thuật toán AI: Isolation Forest phát hiện biến động thân nến/biên độ bất thường ---
    if len(df_window) >= 10:
        # Tính biên độ nến (%) và Độ dài thân nến làm đặc trưng (Features)
        df_window["Body_Change"] = (df_window["Close"] - df_window["Open"]) / df_window["Open"]
        df_window["Spread"] = (df_window["High"] - df_window["Low"]) / df_window["Open"]
        
        X = df_window[["Body_Change", "Spread"]].fillna(0)
        
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(X)
        
        # Label -1 là điểm bất thường (Anomaly)
        df_window["Is_Anomaly"] = [1 if p == -1 else 0 for p in preds]

    # --- Vẽ Giao diện Real-time ---
    with placeholder.container():
        # Metrics Top Bar
        if not df_window.empty:
            latest_candle = df_window.iloc[-1]
            c_open = latest_candle["Open"]
            c_close = latest_candle["Close"]
            diff = c_close - c_open
            pct = (diff / c_open) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Giá BTC/USDT hiện tại", f"${c_close:,.2f}", f"{diff:+,.2f} USDT ({pct:+.2f}%)")
            col2.metric("Tổng số nến đã tạo", len(st.session_state.df_candles))
            
            anomalies_cnt = df_window["Is_Anomaly"].sum() if "Is_Anomaly" in df_window else 0
            col3.metric("Nến bất thường (Window)", int(anomalies_cnt), delta_color="inverse")
            col4.metric("Chu kỳ Nến", f"{candle_duration}s / cây")

        # --- Tạo Biểu đồ Nến Nhật (Candlestick Plotly) ---
        fig = go.Figure()

        # 1. Vẽ các cây nến
        fig.add_trace(go.Candlestick(
            x=df_window["Timestamp"],
            open=df_window["Open"],
            high=df_window["High"],
            low=df_window["Low"],
            close=df_window["Close"],
            name="BTC/USDT OHLC",
            increasing_line_color='#26a69a', # Màu xanh chuẩn sàn trading
            decreasing_line_color='#ef5350' # Màu đỏ chuẩn sàn trading
        ))

        # 2. Đánh dấu các Cây Nến Bất thường (Anomaly Indicator)
        if "Is_Anomaly" in df_window:
            anomalies = df_window[df_window["Is_Anomaly"] == 1]
            if not anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=anomalies["Timestamp"],
                    y=anomalies["High"] * 1.0001, # Hiển thị icon phía trên đỉnh nến
                    mode="markers+text",
                    name="Cảnh báo Anomaly",
                    text=["⚠️ Anomaly"] * len(anomalies),
                    textposition="top center",
                    marker=dict(color="#ff9800", size=10, symbol="triangle-down")
                ))

        # Tùy chỉnh giao diện biểu đồ chuẩn Style TradingView
        fig.update_layout(
            title=f"Biểu đồ Nến Nhật BTC/USDT ({candle_duration}s/nến) - Real-time Binance Stream",
            yaxis_title="Giá (USDT)",
            xaxis_title="Thời gian",
            xaxis_rangeslider_visible=False, # Ẩn slider phụ bên dưới cho gọn
            template="plotly_dark", # Giao diện tối chuyên nghiệp
            height=520,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Xem bảng dữ liệu nến OHLC chi tiết
        with st.expander("📊 Bảng dữ liệu chi tiết các Cây Nến (OHLC Log)"):
            st.dataframe(
                df_window.sort_values(by="Timestamp", ascending=False),
                column_config={
                    "Open": st.column_config.NumberColumn(format="$%.2f"),
                    "High": st.column_config.NumberColumn(format="$%.2f"),
                    "Low": st.column_config.NumberColumn(format="$%.2f"),
                    "Close": st.column_config.NumberColumn(format="$%.2f"),
                },
                use_container_width=True
            )

    # Đợi 1 giây trước khi lấy giá mới
    time.sleep(1)
