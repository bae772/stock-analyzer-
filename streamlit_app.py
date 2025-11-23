import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="간단 주식 분석 앱", layout="wide")
st.title("📊 간단 주식 분석 앱 (미니차트 포함)")

# 사이드바 입력
ticker = st.sidebar.text_input("티커 입력", value="AAPL")
period = st.sidebar.selectbox("기간 선택", ["1mo", "3mo", "6mo", "1y", "2y", "5y"])
interval = st.sidebar.selectbox("인터벌 선택", ["1d", "1h", "30m", "15m"])

if ticker:
    df = yf.download(ticker, period=period, interval=interval)

    if df.empty:
        st.error("데이터를 불러오지 못했습니다. 티커를 확인하세요.")
    else:
        st.subheader(f"{ticker} 가격 데이터")
        st.dataframe(df.tail())

        # ----------------------------
        # 🔹 메인 캔들차트
        # ----------------------------
        fig_main = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        )])
        fig_main.update_layout(title=f"{ticker} 주요 차트")
        st.plotly_chart(fig_main, use_container_width=True)

        # ----------------------------
        # 🔹 미니차트 (미니 라인차트)
        # ----------------------------
        st.subheader("미니 차트 (스파크라인 스타일)")

        mini = go.Figure()
        mini.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(width=2)
        ))

        mini.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )

        st.plotly_chart(mini, use_container_width=True)
