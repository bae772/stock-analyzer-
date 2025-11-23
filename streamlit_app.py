# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ────── 전체 UI 깔끔하게 + 코드 숨기기 ──────
st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")
st.markdown("""
<style>
    .big-font {font-size:50px !important; font-weight:bold; text-align:center;}
    .css-1d391kg, pre, code {display:none !important;}
    .stPlotlyChart {background-color:#000; border-radius:15px; padding:10px;}
    section[data-testid="stSidebar"] {background-color:#111;}
</style>
""", unsafe_allow_html=True)

st.title("키움증권 스타일 AI 분석 카드")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR", label_visibility="collapsed").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if len(data) < 50:
            st.error("데이터 부족")
            st.stop()

        data = data[['Close','High','Low','Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        # ────── 계산 로직 (기존 그대로) ──────
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close-prev_close)/prev_close*100, 2)

        # RSI
        delta = close.diff()
        gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        last_rsi = float(100 - 100/(1+rs.iloc[-1]))

        # 거래량 비율
        vol_ratio = round(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1], 2)

        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]

        # AI 스코어
        score = 50 + max(0,30-last_rsi)*1.4 + change
