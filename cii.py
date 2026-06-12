import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
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

try:
    df_raw = load_and_process_data()
    
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

    # --- BIỂU ĐỒ KỸ THUẬT TƯƠNG TÁC ---
    st.subheader("📈 Biểu đồ Xu hướng Kỹ thuật Toàn cảnh")
    fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['close'], mode='lines', name='Giá đóng cửa', line=dict(color='#1f77b4')), row=1, col=1)
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['MA20'], name='MA(20)', line=dict(color='#ff7f0e', width=1.5)), row=1, col=1)
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['RSI14'], name='RSI(14)', line=dict(color='#9467bd')), row=2, col=1)
    fig_chart.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig_chart.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig_chart.update_layout(xaxis_rangeslider_visible=False, height=400, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_chart, use_container_width=True)

    # --- BẢNG DỮ LIỆU LỊCH SỬ ---
    st.subheader("📋 Nhật ký giao dịch 5 phiên gần đây")
    st.dataframe(df_raw[['time', 'close', 'RSI14', 'MA20']].tail(5).style.format({
        'close': '{:,.0f}', 'RSI14': '{:.2f}', 'MA20': '{:,.0f}'
    }), use_container_width=True)

    st.markdown("---")

    # --- KHỐI DỰ ĐOÁN AI (RANDOM FOREST - SIÊU NHẸ) ---
    st.subheader("🔮 Dự đoán xu hướng giá bằng Trí tuệ nhân tạo (Machine Learning)")
    
    if st.button("🚀 Kích hoạt AI dự đoán giá ngày mai"):
        with st.spinner("Mô hình đang phân tích dữ liệu, vui lòng đợi vài giây..."):
            
            # Chuẩn bị dữ liệu dạng chuỗi thời gian (60 ngày trước đoán ngày sau)
            prices = df_raw['close'].values
            X, y = [], []
            for i in range(60, len(prices)):
                X.append(prices[i-60:i])
                y.append(prices[i])
            X, y = np.array(X), np.array(y)
            
            # Chia tập dữ liệu huấn luyện (80% train)
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]
            
            # Khởi tạo và huấn luyện mô hình Random Forest (Chỉ mất < 1 giây)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Dự đoán trên tập Test
            predictions = model.predict(X_test)
            
            # Dự đoán giá ngày mai
            last_60_days = prices[-60:].reshape(1, -1)
            tomorrow_price = model.predict(last_60_days)[0]
            
            price_delta = tomorrow_price - latest_close
            percent_delta = (price_delta / latest_close) * 100

            # Hiển thị kết quả AI
            st.markdown("### 📈 KẾT QUẢ DỰ ĐOÁN CHO PHIÊN TIẾP THEO")
            col_res1, col_res2 = st.columns([1, 2])
            
            with col_res1:
                if price_delta > 0:
                    st.metric(label="Dự đoán giá phiên ngày mai", value=f"{tomorrow_price:,.0f} VND", delta=f"TĂNG +{price_delta:,.0f} VND (+{percent_delta:.2f}%)", delta_color="normal")
                    st.success("🟢 Tín hiệu ngắn hạn: **TĂNG GIÁ**")
                else:
                    st.metric(label="Dự đoán giá phiên ngày mai", value=f"{tomorrow_price:,.0f} VND", delta=f"GIẢM {price_delta:,.0f} VND ({percent_delta:.2f}%)", delta_color="inverse")
                    st.error("🔴 Tín hiệu ngắn hạn: **GIẢM GIÁ**")
                    
                st.info("⚠️ *Khuyến nghị:* Kết quả dựa trên học máy thống kê lịch sử giá, không bao gồm các yếu tố tin tức vĩ mô đột xuất.")

            with col_res2:
                # Biểu đồ kết quả dự đoán
                fig_ai = go.Figure()
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][:train_size+60], y=df_raw['close'][:train_size+60], name='Dữ liệu Train', line=dict(color='#1f77b4')))
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][train_size+60:], y=y_test, name='Giá Thực Tế (Test)', line=dict(color='#2ca02c')))
                fig_ai.add_trace(go.Scatter(x=df_raw['time'][train_size+60:], y=predictions, name='Đường AI dự báo', line=dict(color='#d62728', dash='dash')))
                fig_ai.update_layout(title='Biểu đồ kiểm chứng thuật toán AI', xaxis_title='Thời gian', yaxis_title='Giá (VND)', height=350)
                st.plotly_chart(fig_ai, use_container_width=True)

except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống dữ liệu: {e}")