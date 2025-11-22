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
        # 1년치 일봉 데이터
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        # NaN 제거
        data = data.dropna()

        close = data['Close']
        high  = data['High']
        low   = data['Low']
        volume = data['Volume']

        # 최신값
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14, min_periods=14).mean()
        avg_loss = loss.rolling(14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        last_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50

        # 이동평균값 안전하게 float 변환
        ma20 = float(close.rolling(20, min_periods=20).mean().iloc[-1])
        ma50 = float(close.rolling(50, min_periods=50).mean().iloc[-1])

        # 거래량 비율
        vol_ma20 = float(volume.rolling(20, min_periods=20).mean().iloc[-1])
        vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1

        # AI 스코어 계산 (키움 느낌)
        score = 50.0
        score += max(0, (30 - last_rsi)) * 1.3
        score += change_pct * 2.5
        score += max(0, (vol_ratio - 1)) * 10
        score += 12 if last_close > ma20 else -8
        score += 10 if last_close > ma50 else -5
        score = int(np.clip(score, 0, 100))

        # 등급 및 색상
        if score >= 80:
            grade, gcolor = "A (강력매수)", "#00ff00"
        elif score >= 70:
            grade, gcolor = "A (매수)", "#33ff33"
        elif score >= 60:
            grade, gcolor = "B (관망)", "#ffff33"
        elif score >= 40:
            grade, gcolor = "C (주의)", "#ff9933"
        else:
            grade, gcolor = "D (매도)", "#ff3333"

        # ATR 기반 목표가 & 손절가
        tr = pd.DataFrame({
            'tr1': high - low,
            'tr2': abs(high - close.shift(1)),
            'tr3': abs(low - close.shift(1))
        }).max(axis=1)
        atr = float(tr.rolling(14, min_periods=14).mean().iloc[-1])
        target_price = round(last_close + atr * 2.5, 2)
        stop_price   = round(last_close - atr * 1.8, 2)
        target_pct   = round((target_price - last_close) / last_close * 100, 1)
        stop_pct     = round((stop_price - last_close) / last_close * 100, 1)

        # 키움 스타일 카드 출력
        st.markdown(f"""
        <div style="background:#000; padding:25px; border-radius:18px; text-align:center; 
                    border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc80; margin:20px 0;">
            <h1 style="color:#00ffcc; margin:0; font-size:4em;">{ticker}</h1>
            <h2 style="color:white; margin:8px 0; font-size:2.8em;">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.6em; margin:5px;">
                {'+' if change_pct>=0 else ''}{change_pct:.2f}%
            </p>

            <h3 style="color:#ccc; margin:20px 0 5px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:6em; margin:0; text-shadow:0 0 20px;">
                {score}
            </h1>

            <h3 style="color:{gcolor}; font-size:2em; margin:15px 0;">
                등급 [{grade}]
            </h3>

            <div style="display:flex; justify-content:space-around; color:white; margin:20px 0; font-size:1.2em;">
                <div>추세: <span style="color:#33ff33;">상승장</span></div>
                <div>캔들: <span style="color:#ffff33;">{'양봉' if change_pct>=0 else '음봉'}</span></div>
            </div>

            <div style="color:#00ffcc; font-size:1.4em; margin:15px 0;">
                거래량: <b>{vol_ratio:.1f}배</b> 
                {'<span style="color:#00ff00;">(폭발!)</span>' if vol_ratio>=3 else ''}
            </div>

            <div style="display:flex; justify-content:space-around; gap:20px; margin:30px 0;">
                <div style="background:#002200; padding:18px; border-radius:12px; flex:1;">
                    <p style="color:#00ff88; margin:0;">TARGET</p>
                    <h3 style="color:#00ff00; margin:8px 0;">${target_price}<br><small>(+{target_pct}%)</small></h3>
                </div>
                <div style="background:#220000; padding:18px; border-radius:12px; flex:1;">
                    <p style="color:#ff8888; margin:0;">STOP LOSS</p>
                    <h3 style="color:#ff3333; margin:8px 0;">${stop_price}<br><small>({stop_pct}%)</small></h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} | 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
