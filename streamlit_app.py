# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ────── UI 설정 + 코드 절대 안 보이게 ──────
st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")

st.markdown("""
<style>
    .css-1d391kg, pre, code, .stAlert {display: none !important;}
    .block-container {padding-top: 2rem;}
    .stPlotlyChart {background: #000; border-radius: 18px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("키움증권 스타일 AI 분석 카드")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR", label_visibility="collapsed").upper()

if ticker:
    try:
        # 데이터 다운로드
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if len(data) < 50:
            st.error("데이터가 부족합니다.")
            st.stop()

        data = data[['Close', 'High', 'Low', 'Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        # ────── 지표 계산 ──────
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss
        last_rsi = float(100 - 100 / (1 + rs.iloc[-1]))

        # 거래량 비율
        vol_ratio = round(float(volume.iloc[-1]) / float(volume.rolling(20).mean().iloc[-1]), 2)

        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])

        # AI 스코어
        score = 50.0
        score += max(0, 30 - last_rsi) * 1.4
        score += change_pct * 2.0
        score += max(0, vol_ratio - 1) * 12
        score += 15 if last_close > ma20 else -10
        score += 10 if last_close > ma50 else -8
        score = int(np.clip(score, 0, 100))

        # 등급
        if score >= 80:
            grade, color = "A (강력매수)", "#00ff00"
        elif score >= 70:
            grade, color = "A (매수)", "#33ff33"
        elif score >= 60:
            grade, color = "B (관망)", "#ffff33"
        elif score >= 40:
            grade, color = "C (주의)", "#ff9933"
        else:
            grade, color = "D (매도)", "#ff3333"

        # ATR 기반 목표가 & 손절가
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        target = round(last_close + atr * 2.5, 2)
        stop   = round(last_close - atr * 1.8, 2)
        target_pct = round((target - last_close) / last_close * 100, 1)
        stop_pct   = round((stop - last_close) / last_close * 100, 1)

        # ────── 키움 카드 출력 ──────
        st.markdown(f"""
        <div style="background:#000;color:#fff;padding:40px;border-radius:25px;text-align:center;
                    border:3px solid #00ffcc;box-shadow:0 0 40px #00ffccaa;margin:25px 0;">
            <h1 style="color:#00ffcc;margin:0;font-size:4.8em">{ticker}</h1>
            <h2 style="margin:15px 0;font-size:3.5em">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'};font-size:2.2em;margin:10px">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>

            <h3 style="color:#aaa;margin:35px 0 10px;font-size:1.8em">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:8em;margin
