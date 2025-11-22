# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="간단 주식 대시보드", layout="wide")
st.title("📊 간단 주식 분석 앱 (BB 제거 버전)")

ticker = st.text_input("주식 티커 입력 (예: AAPL, TSLA, VOO)")
days = st.number_input("최근 추세 기간 (일)", min_value=1, max_value=30, value=5)

if ticker:
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            st.warning("데이터를 가져올 수 없습니다.")
        else:
            data['Close'].fillna(method='bfill', inplace=True)

            # 이동평균
            data['MA20'] = data['Close'].rolling(20).mean().fillna(method='bfill')
            data['MA50'] = data['Close'].rolling(50).mean().fillna(method='bfill')

            # RSI 계산
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean().fillna(method='bfill')
            avg_loss = loss.rolling(14).mean().fillna(method='bfill')
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # 매수/매도 신호 (RSI 기준)
            data['Signal'] = "관망"
            data.loc[data['RSI'] < 30, 'Signal'] = "매수"
            data.loc[data['RSI'] > 70, 'Signal'] = "매도"

            # TXT 다운로드
            recent_data = data[['Close','MA20','MA50','RSI','Signal']].tail(20)
            txt_data = recent_data.to_csv(sep='\t')
            st.download_button(label="TXT 다운로드", data=txt_data, file_name=f"{ticker}_recent.txt", mime="text/plain")

            # 차트
            st.subheader(f"{ticker} 종가 + 신호")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50'))
            buy = data[data['Signal']=='매수']
            sell = data[data['Signal']=='매도']
            fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='매수',
                                     marker=dict(color='green', size=10, symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='매도',
                                     marker=dict(color='red', size=10, symbol='triangle-down')))
            st.plotly_chart(fig, use_container_width=True)

            # 테이블
            st.subheader("최근 데이터")
            st.dataframe(recent_data)

    except Exception as e:
        st.error(f"오류 발생: {e}")
