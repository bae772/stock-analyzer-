# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")

st.title("🔥 키움증권 스타일 AI 분석 카드 (미국주식 전용)")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        # 데이터 다운로드
        data = yf.download(ticker, period="1y", interval="1d")
        if data.empty:
            st.error("티커를 찾을 수 없거나 데이터가 없습니다.")
            st.stop()

        # 기본 가격
        close = data['Close']
        volume = data['Volume']
        last_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        change_pct = (last_close - prev_close) / prev_close * 100

        # RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        # 거래량 폭발 여부 (최근 20일 평균 대비)
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma20.iloc[-1]

        # 간단 AI 스코어 (0~100) - 키움 느낌으로 만들었음
        score = 50
        score += (30 - last_rsi) * 1.2 if last_rsi < 50 else -(last_rsi - 50) * 0.8   # RSI 낮을수록 +
        score += change_pct * 3                                                       # 당일 상승률
        score += min(vol_ratio - 1, 5) * 8 if vol_ratio > 1 else -10                    # 거래량 폭발
        score += 15 if close.iloc[-1] > close.rolling(20).mean().iloc[-1] else -10    # 20일선 위
        score = max(0, min(100, round(score)))

        # 등급
        if score >= 80:   grade, grade_color = "A (강력매수)", "#00ff00"
        elif score >= 70: grade, grade_color = "A (매수)", "#33ff33"
        elif score >= 60: grade, grade_color = "B (관망)", "#ffff33"
        elif score >= 40: grade, grade_color = "C (주의)", "#ff9933"
        else:             grade, grade_color = "D (매도)", "#ff3333"
