# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ────── UI 깔끔 + 코드 절대 안 보이게 ──────
st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")
st.markdown("""
<style>
    .css-1d391kg, pre, code {display:none !important;}
    .stPlotlyChart {background:#111; border-radius:15px; padding:10px;}
    section[data-testid="stSidebar"] {background:#000;}
    .block-container {padding-top:2rem;}
</style>
""", unsafe_allow_html=True)

st.title("베짱이 ai 분석기")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR", label_visibility="collapsed").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if len(data) < 50:
            st.error("데이터 부족")
            st.stop()

        data = data[['Close','High','Low','Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        # ────── 핵심 지표 계산 ──────
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

        # ────── AI 스코어 계산 (완전 수정됨) ──────
        score = 50.0
        score += max(0, 30 - last_rsi) * 1.4          # RSI 낮을수록 +
        score += change_pct * 2.0                     # 당일 등락률
        score += max(0, vol_ratio - 1) * 12           # 거래량 폭발
        score += 15 if last_close > ma20 else -10     # 20일선 위
        score += 10 if last_close > ma50 else -8      # 50일선 위
        score = int(np.clip(score, 0, 100))

        # 등급
        if score >= 80:   grade, color = "A (강력매수)", "#00ff00"
        elif score >= 70: grade, color = "A (매수)", "#33ff33"
        elif score >= 60: grade, color = "B (관망)", "#ffff33"
        elif score >= 40: grade, color = "C (주의)", "#ff9933"
        else:             grade, color = "D (매도)", "#ff3333"

        # ATR 기반 목표가/손절가
        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        target = round(last_close + atr*2.5, 2)
        stop   = round(last_close - atr*1.8, 2)
        target_pct = round((target-last_close)/last_close*100, 1)
        stop_pct   = round((stop-last_close)/last_close*100, 1)

        # ────── 키움 카드 ──────
        st.markdown(f"""
        <div style="background:#000;color:#fff;padding:35px;border-radius:22px;text-align:center;
                    border:3px solid #00ffcc;box-shadow:0 0 40px #00ffcc99;margin:20px 0;">
            <h1 style="color:#00ffcc;margin:0;font-size:4.5em">{ticker}</h1>
            <h2 style="margin:12px 0;font-size:3.2em">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'};font-size:2em">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>

            <h3 style="color:#aaa;margin:30px 0 8px">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:7.5em;margin:0;text-shadow:0 0 35px">{score}</h1>
            <h2 style="color:{color};margin:25px 0;font-size:1.8em">등급 [{grade}]</h2>

            <div style="color:#00ffcc;font-size:1.6em;margin:25px 0">
                거래량 <b>{vol_ratio:.1f}배</b> {'<span style="color:#00ff00">(폭발!)</span>' if vol_ratio>=3 else ''}
            </div>

            <div style="display:flex;justify-content:center;gap:40px;margin:35px 0">
                <div style="background:#002200;padding:22px;border-radius:16px;min-width:160px">
                    <div style="color:#00ff88">TARGET</div>
                    <div style="color:#00ff00;font-size:2em">${target}<br><small>+{target_pct}%</small></div>
                </div>
                <div style="background:#220000;padding:22px;border-radius:16px;min-width:160px">
                    <div style="color:#ff8888">STOP LOSS</div>
                    <div style="color:#ff3333;font-size:2em">${stop}<br><small>{stop_pct}%</small></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ────── 예쁜 차트 ──────
        st.markdown("### 주가 · 이동평균선 · 거래량 차트")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=close, name="종가", line=dict(color="#00ffcc", width=3)))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(20).mean(), name="20일선", line=dict(color="#33ff33")))
        fig.add_trace(go.Scatter(x=data.index, y=close.rolling(50).mean(), name="50일선", line=dict(color="#ff9933")))

        # 거래량 색상 (양봉/음봉)
        vol_colors = ['#00ff88' if c >= o else '#ff4444' 
                      for c, o in zip(close, close.shift(1).fillna(close))]
        fig.add_trace(go.Bar(x=data.index, y=volume, name="거래량", marker_color=vol_colors, opacity=0.3, yaxis="y2"))

        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor="#000",
            paper_bgcolor="#000",
            height=560,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="가격 ($)", showgrid=False),
            yaxis2=dict(title="거래량", overlaying="y", side="right", showgrid=False),
            hovermode="x unified",
            margin=dict(l=20, r=
