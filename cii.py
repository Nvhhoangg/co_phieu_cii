import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from vnstock3 import Vnstock

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Phân tích cổ phiếu CII", layout="wide")
st.title("📊 Hệ thống Phân tích & Dự đoán Giá Cổ phiếu CII")
st.markdown("---")

# --- HÀM TẢI DỮ LIỆU & TÍNH CHỈ BÁO ---
@st.cache_data
def load_and_process_data():
    # Tải dữ liệu từ Vnstock
    stock = Vnstock().stock(symbol='CII', source='VCI')
    df = stock.quote.history(start='2023-01-01', end='2026-06-11')
    df = df.sort_values('time').reset_index(drop=True)
    
    # Đảm bảo định dạng ngày tháng và số
    df['time'] = pd.to_datetime(df['time'])
    
    # 1. Tính toán Đường trung bình động MA(20)
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 2. Tính toán Chỉ số sức mạnh tương đối RSI(14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    return df

# Hàm train mô hình được cache để tránh việc người dùng bấm nút là phải train lại từ đầu
@st.cache_resource
def train_lstm_model(scaled_data, train_size):
    train_data = scaled_data[0:train_size, :]
    x_train, y_train = [], []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)
    return model

try:
    df_raw = load_and_process_data()
    data = df_raw[['time', 'close']].set_index('time')
    
    # --- KHỐI THÔNG TIN TỔNG QUAN TẠI THỜI ĐIỂM HIỆN TẠI ---
    st.subheader("📌 Chỉ số kỹ thuật phiên gần nhất")
    latest_row = df_raw.iloc[-1]
    latest_close = latest_row['close']
    latest_rsi = latest_row['RSI14']
    latest_ma20 = latest_row['MA20']
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric(label="Giá đóng cửa gần nhất", value=f"{latest_close:,.0f} VND")
    with col_info2:
        rsi_status = "Quá mua (Rủi ro)" if latest_rsi > 70 else ("Quá bán (Cơ hội)" if latest_rsi < 30 else "Trung tính")
        st.metric(label="Chỉ báo RSI (14)", value=f"{latest_rsi:.2f}", delta=rsi_status, delta_color="off" if rsi_status == "Trung tính" else "inverse")
    with col_info3:
        ma_delta = f"Trên MA20 (+{latest_close - latest_ma20:,.0f}đ)" if latest_close > latest_ma20 else f"Dưới MA20 ({latest_close - latest_ma20:,.0f}đ)"
        st.metric(label="Đường MA(20)", value=f"{latest_ma20:,.0f} VND", delta=ma_delta)

    st.markdown("---")

    # --- BỔ SUNG 1: BIỂU ĐỒ KỸ THUẬT TƯƠNG TÁC (CHƯA CẦN BẤM AI) ---
    st.subheader("📈 Biểu đồ Xu hướng Kỹ thuật Toàn cảnh")
    
    # Tạo subplot: Trên là đồ thị Nến + MA20, Dưới là chỉ báo RSI
    fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Vẽ nến (Dùng tạm giá close nếu nguồn VCI không đủ open/high/low, nhưng thông thường vnstock có đủ)
    # Giả định dữ liệu có open, high, low, close. Nếu không có, thay bằng line chart của 'close'
    if all(col in df_raw.columns for col in ['open', 'high', 'low', 'close']):
        fig_chart.add_trace(go.Candlestick(x=df_raw['time'], open=df_raw['open'], high=df_raw['high'],
                                           low=df_raw['low'], close=df_raw['close'], name='Nến giá'), row=1, col=1)
    else:
        fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['close'], mode='lines', name='Giá đóng cửa', line=dict(color='#1f77b4')), row=1, col=1)
        
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['MA20'], name='MA(20)', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
    
    # Vẽ RSI
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['RSI14'], name='RSI(14)', line=dict(color='#9467bd')), row=2, col=1)
    fig_chart.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_chart.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig_chart.update_layout(xaxis_rangeslider_visible=False, height=500, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_chart, use_container_width=True)

    # --- BỔ SUNG 2: BẢNG DỮ LIỆU LỊCH SỬ GẦN NHẤT ---
    st.subheader("📋 Nhật ký giao dịch 5 phiên gần đây")
    display_cols = [c for c in ['time', 'open', 'high', 'low', 'close', 'volume', 'RSI14', 'MA20'] if c in df_raw.columns]
    st.dataframe(df_raw[display_cols].tail(5).style.format({
        'open': '{:,.0f}', 'high': '{:,.0f}', 'low': '{:,.0f}', 'close': '{:,.0f}', 
        'volume': '{:,.0f}', 'RSI14': '{:.2f}', 'MA20': '{:,.0f}'
    }), use_container_width=True)

    st.markdown("---")

    # --- KHỐI DỰ ĐOÁN LSTM ---
    st.subheader("🔮 Dự đoán xu hướng giá bằng Trí tuệ nhân tạo (LSTM)")
    
    if st.button("🚀 Kích hoạt AI dự đoán giá ngày mai"):
        with st.spinner("Mô hình đang phân tích chuỗi thời gian, vui lòng đợi..."):
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            train_size = int(len(scaled_data) * 0.8)
            
            # Gọi hàm train đã được tối ưu cache
            model = train_lstm_model(scaled_data, train_size)

            # Dự đoán trên tập kiểm thử
            test_data = scaled_data[train_size - 60:, :]
            x_test = []
            for i in range(60, len(test_data)):
                x_test.append(test_data[i-60:i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)

            # Dự đoán tương lai
            last_60_days = scaled_data[-60:]
            x_future = np.array([last_60_days])
            x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))
            
            future_prediction = model.predict(x_future)
            tomorrow_price = scaler.inverse_transform(future_prediction)[0][0]
            
            price_delta = tomorrow_price - latest_close
            percent_delta = (price_delta / latest_close) * 100

            # Hiển thị kết quả AI
            st.markdown("### 📈 KẾT QUẢ DỰ ĐOÁN CHO PHIÊN TIẾP THEO")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if price_delta > 0:
                    st.metric(
                        label="Dự đoán giá phiên ngày mai", 
                        value=f"{tomorrow_price:,.0f} VND", 
                        delta=f"TĂNG +{price_delta:,.0f} VND (+{percent_delta:.2f}%)",
                        delta_color="normal"
                    )
                    st.success("🟢 Tín hiệu ngắn hạn: **TĂNG GIÁ**")
                else:
                    st.metric(
                        label="Dự đoán giá phiên ngày mai", 
                        value=f"{tomorrow_price:,.0f} VND", 
                        delta=f"GIẢM {price_delta:,.0f} VND ({percent_delta:.2f}%)",
                        delta_color="inverse"
                    )
                    st.error("🔴 Tín hiệu ngắn hạn: **GIẢM GIÁ**")
                    
                st.info("⚠️ *Khuyến nghị:* Kết quả được tính toán dựa trên thuật toán kỹ thuật học máy lịch sử giá, không bao gồm tin tức vĩ mô hoặc các sự kiện bất khả kháng.")

            with col_res2:
                # Đổi biểu đồ Matplotlib sang Plotly cho đồng bộ và mượt mà hơn
                fig_ai = go.Figure()
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][:train_size], y=df_raw['close'][:train_size], name='Dữ liệu Train', line=dict(color='#1f77b4')))
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][train_size:], y=df_raw['close'][train_size:], name='Giá Thực Tế (Test)', line=dict(color='#2ca02c')))
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][train_size:], y=predictions.flatten(), name='Đường AI dự báo', line=dict(color='#d62728', dash='dash')))
                
                fig_ai.update_layout(title='Biểu đồ so sánh độ chính xác của Mô hình AI', xaxis_title='Thời gian', yaxis_title='Giá (VND)', height=400)
                st.plotly_chart(fig_ai, use_container_width=True)

except Exception as e:
    st.error(f"Đã xảy ra lỗi khi đồng bộ dữ liệu hoặc xử lý: {e}")