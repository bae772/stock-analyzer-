# streamlit_app.py (최종 완성본 - 코드 절대 안 보임 버전)
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="키움 AI 카드", layout="centered")

# 이 두 줄이 핵심! 코드 안 보이게 하는 마법
st.markdown("<style>pre, code {display: none !important;}</style>", unsafe_allow_html=True)
st.markdown("<style>body {background-color:#000;}</style>", unsafe_allow_html=True)

ticker = st.text_input("티커 입력", value="BMR").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False)
        if len(data) < 50: raise Exception("데이터 부족")

        close = data['Close'].dropna()
        high  = data['High']
        low   = data['Low']
        vol   = data['Volume']

        last  = float(close.iloc[-1])
        prev  = float(close.iloc[-2])
        chg   = round((last-prev)/prev*100, 2)
        rsi   = float((100 - 100/(1 + (close.diff().clip(lower=0).rolling(14).mean() / 
                     abs(close.diff().clip(upper=0)).rolling(14).mean()))).iloc[-1])
        vol_r = round(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1], 2)

        score = int(np.clip(50 + max(0,30-rsi)*1.4 + chg*2 + max(0,vol_r-1)*12 + 
                           (15 if last > close.rolling(20).mean().iloc[-1] else -10), 0, 100))

        grade = ["D (매도)","C (주의)","B (관망)","A (매수)","A (강력매수)"][min(4, score//20)]
        color = ["#ff3333","#ff9933","#ffff33","#33ff33","#00ff00"][min(4, score//20)]

        tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        target = round(last + atr*2.5, 2)
        stop   = round(last - atr*1.8, 2)

        st.markdown(f"""
        <div style="background:#000;color:#fff;padding:30px;border-radius:20px;
                    border:3px solid #00ffcc;box-shadow:0 0 30px #00ffcc80;text-align:center">
            <h1 style="color:#00ffcc;margin:0;font-size:4em">{ticker}</h1>
            <h2 style="margin:10px 0;font-size:3em">${last:.2f}</h2>
            <p style="color:{'#33ff33' if chg>=0 else '#ff3333'};font-size:2em;margin:5px">
                {'+' if chg>=0 else ''}{chg}%
            </p>
            <h3 style="color:#aaa;margin:25px 0 5px">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:7em;margin:0">{score}</h1>
            <h2 style="color:{color};margin:20px 0">등급 [{grade}]</h2>
            <div style="color:#00ffcc;font-size:1.5em;margin:20px 0">
                거래량 <b>{vol_r:.1f}배</b> {'<span style="color:#00ff00">(폭발!)</span>' if vol_r>=3 else ''}
            </div>
            <div style="display:flex;justify-content:center;gap:30px;margin:30px 0">
                <div style="background:#002200;padding:20px;border-radius:15px">
                    <div style="color:#00ff88">TARGET</div>
                    <div style="color:#00ff00;font-size:1.8em">${target}<br><small>+{round((target-last)/last*100,1)}%</small></div>
                </div>
                <div style="background:#220000;padding:20px;border-radius:15px">
                    <div style="color:#ff8888">STOP LOSS</div>
                    <div style="color:#ff3333;font-size:1.8em">${stop}<br><small>{round((stop-last)/last*100,1)}%</small></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트 {datetime.now().strftime('%m-%d %H:%M')}")

    except Exception as e:
        st.error("데이터를 불러올 수 없어요")
