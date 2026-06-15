import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
# Import module Quote chính thức của vnstock3
from vnstock import Quote 

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Phân tích cổ phiếu CII", layout="wide")
st.title("📊 Hệ thống Phân tích & Dự đoán Giá Cổ phiếu CII")
st.markdown("---")

# --- HÀM TẢI DỮ LIỆU THỰC TỪ VNSTOCK ---
@st.cache_data(ttl=600) # Chỉ lưu cache 10 phút để cập nhật dữ liệu mới liên tục
def load_and_process_data():
    df = None
    # Thử kết nối trực tiếp với nguồn TCBS - nguồn trả dữ liệu OHLC chuẩn nhất hiện tại
    try:
        quote = Quote(symbol='CII', source='TCBS')
        # Lấy dữ liệu lịch sử từ đầu năm 2023 đến nay
        df = quote.history(start='2023-01-01', end='2026-06-15', interval='1D')
    except Exception as e:
        # Nếu TCBS nghẽn, tự động thử nguồn VCI làm dự phòng
        try:
            quote = Quote(symbol='CII', source='VCI')
            df = quote.history(start='2023-01-01', end='2026-06-15', interval='1D')
        except Exception:
            df = None

    # Nếu lấy được dữ liệu thực thành công
    if df is not None and not df.empty:
        st.sidebar.success("✅ Đang hiển thị: DỮ LIỆU THỰC TẾ THỊ TRƯỜNG")
        
        # Đảm bảo tên cột thời gian đồng nhất
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'time'})
            
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        
        # Ép kiểu dữ liệu số
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # LƯU Ý: Nếu dữ liệu vnstock trả về bị chia cho 1000 (đơn vị nghìn đồng), 
        # ta nhân lại với 1000 để hiển thị đúng giá trị thực tế trên bảng điện (Ví dụ: 18,000 VND)
        if df['close'].max() < 1000:
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * 1000
    else:
        # Chỉ khi hoàn toàn mất mạng hoặc API sập hoàn toàn mới dùng đến kho dữ liệu tĩnh này
        st.sidebar.error("⚠️ Lỗi kết nối! Đang hiển thị dữ liệu giả lập dự phòng.")
        dates = pd.date_range(start="2025-01-01", end="2026-06-15", freq="D")
        np.random.seed(42)
        base_price = 18000  # Đặt mức giá nền thực tế cho CII
        prices = base_price + np.cumsum(np.random.normal(5, 300, len(dates)))
        
        df = pd.DataFrame({
            'time': dates, 'close': prices, 
            'open': prices - np.random.randint(-150, 150, len(dates)), 
            'volume': np.random.randint(100000, 5000000, len(dates))
        })
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.randint(0, 200, len(dates))
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.randint(0, 200, len(dates))

    # --- TÍNH TOÁN CÁC CHỈ BÁO KỸ THUẬT ---
    df = df.dropna(subset=['time', 'close']).sort_values('time').reset_index(drop=True)
    
    # 1. Đường MA(20)
    df['MA20'] = df['close'].rolling(window=20).mean()
    
    # 2. Chỉ số RSI(14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    return df

try:
    df_raw = load_and_process_data()
    
    # --- KHỐI THÔNG TIN TỔNG QUAN ---
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
        st.metric(label="Chỉ báo RSI (14)", value=f"{latest_rsi:.2f}" if not pd.isna(latest_rsi) else "100", delta=rsi_status, delta_color="off" if rsi_status == "Trung tính" else "inverse")
    with col_info3:
        if pd.isna(latest_ma20):
            st.metric(label="Đường MA(20)", value="Đang tính toán...")
        else:
            ma_delta = f"Trên MA20 (+{latest_close - latest_ma20:,.0f}đ)" if latest_close > latest_ma20 else f"Dưới MA20 ({latest_close - latest_ma20:,.0f}đ)"
            st.metric(label="Đường MA(20)", value=f"{latest_ma20:,.0f} VND", delta=ma_delta)

    st.markdown("---")

    # --- BIỂU ĐỒ KỸ THUẬT TƯƠNG TÁC (NẾN NHẬT & RSI) ---
    st.subheader("📈 Biểu đồ Xu hướng Kỹ thuật Toàn cảnh")
    
    fig_chart = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # 1. Thêm biểu đồ nến Nhật từ dữ liệu Vnstock thực
    fig_chart.add_trace(go.Candlestick(
        x=df_raw['time'], open=df_raw['open'], high=df_raw['high'], low=df_raw['low'], close=df_raw['close'],
        name='Nến Nhật', increasing_line_color='#26a69a', decreasing_line_color='#ef5350'
    ), row=1, col=1)
    
    # 2. Thêm đường MA(20)
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['MA20'], name='MA(20)', line=dict(color='#ff9800', width=2)), row=1, col=1)
    
    # 3. Thêm đường RSI(14) vào hàng 2
    fig_chart.add_trace(go.Scatter(x=df_raw['time'], y=df_raw['RSI14'], name='RSI(14)', line=dict(color='#9c27b0', width=1.5)), row=2, col=1)
    fig_chart.add_hline(y=70, line_dash="dash", line_color="#ef5350", row=2, col=1)
    fig_chart.add_hline(y=30, line_dash="dash", line_color="#26a69a", row=2, col=1)
    
    fig_chart.update_layout(
        xaxis_rangeslider_visible=False, xaxis2_rangeslider_visible=False,
        height=600, margin=dict(t=20, b=20, l=20, r=20), hovermode='x unified'
    )
    st.plotly_chart(fig_chart, use_container_width=True)

    # --- BẢNG DỮ LIỆU LỊCH SỬ ---
    st.subheader("📋 Nhật ký giao dịch 5 phiên gần đây")
    st.dataframe(df_raw[['time', 'open', 'high', 'low', 'close', 'RSI14', 'MA20']].tail(5).style.format({
        'open': '{:,.0f}', 'high': '{:,.0f}', 'low': '{:,.0f}', 'close': '{:,.0f}', 'RSI14': '{:.2f}', 'MA20': '{:,.0f}'
    }), use_container_width=True)

    st.markdown("---")

    # --- KHỐI DỰ ĐOÁN AI (RANDOM FOREST) ---
    st.subheader("🔮 Dự đoán xu hướng giá bằng Trí tuệ nhân tạo (Machine Learning)")
    
    if st.button("🚀 Kích hoạt AI dự đoán giá ngày mai"):
        with st.spinner("Mô hình đang phân tích dữ liệu thực..."):
            prices = df_raw['close'].values
            if len(prices) > 60:
                X, y = [], []
                for i in range(60, len(prices)):
                    X.append(prices[i-60:i])
                    y.append(prices[i])
                X, y = np.array(X), np.array(y)
                
                train_size = int(len(X) * 0.8)
                X_train, y_train = X[:train_size], y[:train_size]
                X_test, y_test = X[train_size:], y[train_size:]
                
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                last_60_days = prices[-60:].reshape(1, -1)
                tomorrow_price = model.predict(last_60_days)[0]
                
                price_delta = tomorrow_price - latest_close
                percent_delta = (price_delta / latest_close) * 100

                st.markdown("### 📈 KẾT QUẢ DỰ ĐOÁN CHO PHIÊN TIẾP THEO")
                col_res1, col_res2 = st.columns([1, 2])
                
                with col_res1:
                    if price_delta > 0:
                        st.metric(label="Dự đoán giá phiên ngày mai", value=f"{tomorrow_price:,.0f} VND", delta=f"TĂNG +{price_delta:,.0f} VND (+{percent_delta:.2f}%)")
                        st.success("🟢 Tín hiệu ngắn hạn: **TĂNG GIÁ**")
                    else:
                        st.metric(label="Dự đoán giá phiên ngày mai", value=f"{tomorrow_price:,.0f} VND", delta=f"GIẢM {price_delta:,.0f} VND ({percent_delta:.2f}%)", delta_color="inverse")
                        st.error("🔴 Tín hiệu ngắn hạn: **GIẢM GIÁ**")
                
                with col_res2:
                    fig_ai = go.Figure()
                    fig_ai.add_trace(go.Scatter(y=y_test, name='Giá Thực Tế', line=dict(color='#2ca02c')))
                    fig_ai.add_trace(go.Scatter(y=predictions, name='Đường AI dự báo', line=dict(color='#d62728', dash='dash')))
                    fig_ai.update_layout(title='Biểu đồ kiểm chứng thuật toán AI', height=350)
                    st.plotly_chart(fig_ai, use_container_width=True)
            else:
                st.warning("Dữ liệu lịch sử hiện tại chưa đủ 60 phiên để huấn luyện mô hình học máy.")

except Exception as e:
    st.error(f"Đã xảy ra lỗi hệ thống dữ liệu: {e}")