import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from streamlit_autorefresh import st_autorefresh

# ==========================================
# 1. CẤU HÌNH TRANG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Real-Time BTC Candlestick & Anomaly",
    page_icon="📈",
    layout="wide"
)

# Tự động làm mới dữ liệu realtime mỗi 2 giây (2000 ms)
st_autorefresh(interval=2000, key="binance_btc_realtime")

st.title("🕯️ Biểu đồ Nến Nhật BTC/USDT Real-Time & Anomaly Detection")
st.caption("Hệ thống nạp dữ liệu nến thật từ Binance, cập nhật thời gian thực & phát hiện biến động giá bất thường bằng AI")

# ==========================================
# 2. THANH ĐIỀU KHIỂN BÊN HÔNG (SIDEBAR)
# ==========================================
st.sidebar.header("⚙️ Cấu hình Nến & Luồng Dữ liệu")

timeframe = st.sidebar.selectbox(
    "Khung thời gian Nến (Timeframe):",
    options=["1m", "3m", "5m", "15m"],
    index=0
)

window_candles = st.sidebar.slider(
    "Số lượng Nến hiển thị (Window Size):",
    min_value=30,
    max_value=200,
    value=100
)

contamination = st.sidebar.slider(
    "Tỉ lệ nhạy Anomaly (Contamination):",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)

# ==========================================
# 3. HÀM LẤY DỮ LIỆU NẾN THẬT TỪ BINANCE API
# ==========================================
@st.cache_data(ttl=1)  # Cache 1 giây để tối ưu tốc độ
def fetch_binance_klines(symbol="BTCUSDT", interval="1m", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            # Cấu trúc Klines Binance: [Open time, Open, High, Low, Close, Volume, ...]
            df = pd.DataFrame(data, columns=[
                "Open_Time", "Open", "High", "Low", "Close", "Volume",
                "Close_Time", "Quote_Asset_Volume", "Number_of_Trades",
                "Taker_Buy_Base", "Taker_Buy_Quote", "Ignore"
            ])
            
            # Ép kiểu dữ liệu chuẩn
            df["Timestamp"] = pd.to_datetime(df["Open_Time"], unit="ms")
            df["Open"] = df["Open"].astype(float)
            df["High"] = df["High"].astype(float)
            df["Low"] = df["Low"].astype(float)
            df["Close"] = df["Close"].astype(float)
            
            return df[["Timestamp", "Open", "High", "Low", "Close"]]
    except Exception as e:
        pass
    return pd.DataFrame()

# Nạp dữ liệu nến lịch sử + cây nến hiện tại đang nhảy
df_candles = fetch_binance_klines(symbol="BTCUSDT", interval=timeframe, limit=window_candles)

# ==========================================
# 4. MÔ HÌNH AI: ISOLATION FOREST PHÁT HIỆN ANOMALY
# ==========================================
if not df_candles.empty and len(df_candles) >= 10:
    # Đặc trưng 1: % Độ dài thân nến (Close - Open)
    df_candles["Body_Change"] = (df_candles["Close"] - df_candles["Open"]) / df_candles["Open"]
    # Đặc trưng 2: Biên độ dao động lớn nhất trong cây nến (High - Low)
    df_candles["Spread"] = (df_candles["High"] - df_candles["Low"]) / df_candles["Open"]
    
    X = df_candles[["Body_Change", "Spread"]].fillna(0)
    
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(X)
    
    # Label -1 đại diện cho nến Anomaly (biến động bất thường)
    df_candles["Is_Anomaly"] = [1 if p == -1 else 0 for p in preds]

# ==========================================
# 5. HIỂN THỊ CÁC THÔNG SỐ TỔNG QUAN (METRICS)
# ==========================================
if not df_candles.empty:
    latest = df_candles.iloc[-1]
    c_open = latest["Open"]
    c_close = latest["Close"]
    diff = c_close - c_open
    pct = (diff / c_open) * 100 if c_open != 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá BTC/USDT hiện tại", f"${c_close:,.2f}", f"{diff:+,.2f} USDT ({pct:+.2f}%)")
    col2.metric("Nến mở cửa (Open)", f"${c_open:,.2f}")
    
    anomalies_cnt = df_candles["Is_Anomaly"].sum() if "Is_Anomaly" in df_candles else 0
    col3.metric("Nến bất thường (Anomaly)", int(anomalies_cnt), delta_color="inverse")
    col4.metric("Khung Nến / Cập nhật", f"{timeframe} (Mỗi 2s)")

# ==========================================
# 6. VẼ BIỂU ĐỒ NẾN NHẬT TRADINGVIEW (PLOTLY)
# ==========================================
fig = go.Figure()

if not df_candles.empty:
    # 1. Vẽ các Cây Nến (Xanh / Đỏ)
    fig.add_trace(go.Candlestick(
        x=df_candles["Timestamp"],
        open=df_candles["Open"],
        high=df_candles["High"],
        low=df_candles["Low"],
        close=df_candles["Close"],
        name="BTC/USDT OHLC",
        increasing_line_color='#26a69a',  # Màu xanh chuẩn sàn Trading
        decreasing_line_color='#ef5350'   # Màu đỏ chuẩn sàn Trading
    ))

    # 2. Đánh dấu các nến Bất thường bằng Mũi tên Cảnh báo
    if "Is_Anomaly" in df_candles:
        anomalies = df_candles[df_candles["Is_Anomaly"] == 1]
        if not anomalies.empty:
            fig.add_trace(go.Scatter(
                x=anomalies["Timestamp"],
                y=anomalies["High"] * 1.0003,  # Hiện icon ngay phía trên đỉnh nến
                mode="markers+text",
                name="Cảnh báo Anomaly",
                text=["⚠️"] * len(anomalies),
                textposition="top center",
                marker=dict(color="#ff9800", size=10)
            ))

fig.update_layout(
    title=f"Biểu đồ Nến Nhật BTC/USDT ({timeframe}) - Real-time Binance API",
    yaxis_title="Giá (USDT)",
    xaxis_title="Thời gian",
    xaxis_rangeslider_visible=False,
    template="plotly_white",  # Nền trắng sáng đẹp chuyên nghiệp
    height=550,
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

# Xem bảng dữ liệu nến OHLC chi tiết
with st.expander("📊 Bảng dữ liệu chi tiết các Cây Nến (OHLC Log)"):
    st.dataframe(
        df_candles.sort_values(by="Timestamp", ascending=False),
        column_config={
            "Open": st.column_config.NumberColumn(format="$%.2f"),
            "High": st.column_config.NumberColumn(format="$%.2f"),
            "Low": st.column_config.NumberColumn(format="$%.2f"),
            "Close": st.column_config.NumberColumn(format="$%.2f"),
        },
        use_container_width=True
    )
