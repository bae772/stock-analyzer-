import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="종합 주식 분석기", layout="wide")
st.title("📊 종합 기술적 분석 + 매수·매도 신호")

# ===============================
# 지표 계산 함수
# ===============================
def compute_indicators(df):

    # 이동평균선
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()

    # RSI 계산
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 캔들 패턴 계산
    df['Bullish_Engulfing'] = (
        (df['Close'] > df['Open']) &
        (df['Close'].shift(1) < df['Open'].shift(1))
    )

    df['Bearish_Engulfing'] = (
        (df['Close'] < df['Open']) &
        (df['Close'].shift(1) > df['Open'].shift(1))
    )

    df['Hammer'] = (
        ((df['High'] - df['Close']) <= (df['Open'] - df['Low']) * 0.3) &
        ((df['Open'] - df['Low']) >= (df['High'] - df['Open']) * 2)
    )

    df['Shooting_Star'] = (
        ((df['Close'] - df['Low']) <= (df['High'] - df['Open']) * 0.3) &
        ((df['High'] - df['Open']) >= (df['Open'] - df['Low']) * 2)
    )

    # 종합 매수 신호
    df['Buy_Signal'] = (
        (df['RSI'] < 30) |
        (df['Bullish_Engulfing']) |
        (df['Hammer'])
    )

    # 종합 매도 신호
    df['Sell_Signal'] = (
        (df['RSI'] > 70) |
        (df['Bearish_Engulfing']) |
        (df['Shooting_Star'])
    )

    # 신호 이유 자동 생성
    def get_reason(row):
        reasons = []

        if row['Buy_Signal']:
            if row['RSI'] < 30:
                reasons.append("RSI 과매도 (30 이하)")
            if row['Bullish_Engulfing']:
                reasons.append("강한 양봉 장악형 (Bullish Engulfing)")
            if row['Hammer']:
                reasons.append("반등 패턴 Hammer")

        if row['Sell_Signal']:
            if row['RSI'] > 70:
                reasons.append("RSI 과매수 (70 이상)")
            if row['Bearish_Engulfing']:
                reasons.append("강한 음봉 장악형 (Bearish Engulfing)")
            if row['Shooting_Star']:
                reasons.append("반전 패턴 Shooting Star")

        return ", ".join(reasons)

    df["Signal_Reason"] = df.apply(get_reason, axis=1)
    return df


# ===============================
# 사용자 입력
# ===============================
ticker = st.text_input("종목 티커 입력 (예: AAPL, TSLA, NVDA, 005930.KS)", "AAPL")

if st.button("데이터 분석 실행"):
    df = yf.download(ticker, period="6mo")

    if df.empty:
        st.error("데이터 로딩 실패. 티커를 다시 확인하세요.")
        st.stop()

    df = compute_indicators(df)

    # ===========================================
    # 표 출력 (매수/매도 신호 및 이유 포함)
    # ===========================================
    st.subheader("매수·매도 신호 요약")
    st.dataframe(df[["Close", "RSI", "MA20", "MA50",
                     "Buy_Signal", "Sell_Signal", "Signal_Reason"]].tail(20))

    # ===========================================
    # Plotly 차트 생성
    # ===========================================
    fig = go.Figure()

    # 캔들 차트
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Candles"
    ))

    # 이동평균선
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA20"], mode="lines", name="MA20"
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA50"], mode="lines", name="MA50"
    ))

    # 매수 신호 (초록 점)
    fig.add_trace(go.Scatter(
        x=df.index[df["Buy_Signal"]],
        y=df["Close"][df["Buy_Signal"]],
        mode="markers",
        marker=dict(size=10, color="green"),
        name="Buy"
    ))

    # 매도 신호 (빨간 점)
    fig.add_trace(go.Scatter(
        x=df.index[df["Sell_Signal"]],
        y=df["Close"][df["Sell_Signal"]],
        mode="markers",
        marker=dict(size=10, color="red"),
        name="Sell"
    ))

    fig.update_layout(height=700, title=f"{ticker} 기술적 분석 차트")
    st.plotly_chart(fig, use_container_width=True)
