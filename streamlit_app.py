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

        # 목표가 & 손절가 (단순 ATR 기반)
        atr = (data['High'] - data['Low']).rolling(14).mean().iloc[-1]
        target_price = round(last_close + atr * 2.5, 2)
        stop_price = round(last_close - atr * 1.8, 2)
        target_pct = round((target_price - last_close) / last_close * 100, 1)
        stop_pct = round((stop_price - last_close) / last_close * 100, 1)

        # 키움 스타일 검정 카드 레이아웃
        st.markdown(
            f"""
            <div style="
                background-color: #000000;
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                border: 2px solid #00ffcc;
                box-shadow: 0 0 20px #00ffcc;
                margin: 20px 0;
            ">
                <h1 style="color:#00ffcc; margin:0; font-size:3.5em;">{ticker}</h1>
                <h2 style="color:white; margin:5px; font-size:2.5em;">${last_close:.2f}</h2>
                <p style="color:{'#33ff33' if change_pct>0 else '#ff3333'}; font-size:1.3em; margin:5px;">
                    {'+' if change_pct>0 else ''}{change_pct:.2f}%
                </p>

                <h3 style="color:#cccccc; margin:15px 0 5px;">AI SCORE</h3>
                <h1 style="
                    color: {'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                    font-size: 5.5em;
                    margin:0;
                    text-shadow: 0 0 20px;
                ">{score}</h1>

                <h3 style="color:{grade_color}; font-size:1.8em; margin:10px 0;">
                    등급 [{grade.split(' ')[0]} ({grade.split(' ')[1] if len(grade.split())>1 else ''})]
                </h3>

                <div style="display:flex; justify-content:space-around; margin:20px 0; color:white; font-size:1.1em;">
                    <div>추세: <span style="color:#33ff33;">상승장 (강함)</span></div>
                    <div>캔들: <span style="color:#ffff33;">{'양봉' if change_pct>0 else '음봉'}</span></div>
                </div>

                <div style="color:#00ffcc; font-size:1.3em; margin:15px 0;">
                    거래량: <b>{vol_ratio:.1f}배</b> 
                    {'<span style="color:#00ff00;">폭발</span>' if vol_ratio>=3 else '보통'}
                </div>

                <div style="display:flex; justify-content:space-around; margin:30px 0;">
                    <div style="background:#003300; padding:15px; border-radius:10px; width:45%;">
                        <p style="color:#00ff00; margin:0; font-size:1.1em;">TARGET (목표)</p>
                        <h3 style="color:#00ff00; margin:5px;">${target_price} <small>(+{target_pct}%)</small></h3>
                    </div>
                    <div style="background:#330000; padding:15px; border-radius:10px; width:45%;">
                        <p style="color:#ff3333; margin:0; font-size:1.1em;">STOP LOSS (손절)</p>
                        <h3 style="color:#ff3333; margin:5px;">${stop_price} <small>({stop_pct}%)</small></h3>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"마지막 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류: {e}")
