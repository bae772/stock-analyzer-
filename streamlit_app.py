# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")
st.title("키움증권 스타일 AI 분석 카드 (미국주식)")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        # 필요한 열만 사용 + NaN 제거
        data = data[['Close', 'High', 'Low', 'Volume']].dropna()

        close  = data['Close']
        high   = data['High']
        low    = data['Low']
        volume = data['Volume']

        # 스칼라 변환
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI 14
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        last_rsi = float(rsi.iloc[-1])

        # 거래량 비율
        vol_ma20   = float(volume.rolling(20).mean().iloc[-1])
        vol_today  = float(volume.iloc[-1])
        vol_ratio  = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 1.0

        # 20일선, 50일선
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])

        # ────── AI 스코어 계산 ──────
        score = 50.0
        score += max(0, 30 - last_rsi) * 1.4
        score += change_pct * 2.0
        score += max(0, vol_ratio - 1) * 12
        score += 15 if last_close > ma20 else -10
        score += 10 if last_close > ma50 else -8
        score = int(np.clip(score, 0, 100))

        # 등급
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
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low  - close.shift(1))
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        target_price = round(last_close + atr * 2.5, 2)
        stop_price   = round(last_close - atr * 1.8, 2)
        target_pct   = round((target_price - last_close) / last_close * 100, 1)
        stop_pct     = round((stop_price - last_close) / last_close * 100, 1)

        # ────── AI SCORE 카드 출력 ──────
        st.markdown(f"""
        <div style="background:#000; color:white; padding:30px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:4.5em;">{ticker}</h1>
            <h2 style="margin:10px 0; font-size:3em;">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.8em; margin:5px;">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>

            <h3 style="color:#aaa; margin:25px 0 5px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:7em; margin:0; text-shadow:0 0 30px;">
                {score}
            </h1>

            <h2 style="color:{gcolor}; margin:20px 0;">등급 [{grade}]</h2>

            <div style="display:flex; justify-content:center; gap:40px; margin:20px 0; font-size:1.3em;">
                <div>추세 <span style="color:#33ff33;">{'상승장' if last_close > ma20 else '하락장'}</span></div>
                <div>캔들 <span style="color:#ffff33;">{'양봉' if change_pct>=0 else '음봉'}</span></div>
            </div>

            <div style="color:#00ffcc; font-size:1.5em; margin:20px 0;">
                거래량 <b>{vol_ratio:.1f}배</b> 
                {'<span style="color:#00ff00;">(폭발!)</span>' if vol_ratio >= 3 else ''}
            </div>

            <div style="display:flex; justify-content:center; gap:30px; margin:30px 0;">
                <div style="background:#002200; padding:20px; border-radius:15px; min-width:150px;">
                    <div style="color:#00ff88;">TARGET</div>
                    <div style="color:#00ff00; font-size:1.8em;">${target_price}<br><small>+{target_pct}%</small></div>
                </div>
                <div style="background:#220000; padding:20px; border-radius:15px; min-width:150px;">
                    <div style="color:#ff8888;">STOP LOSS</div>
                    <div style="color:#ff3333; font-size:1.8em;">${stop_price}<br><small>{stop_pct}%</small></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ────── AI SCORE 해석 ──────
        if score >= 80:
            interpretation = "강력매수: 상승 가능성이 높고 투자 매력도 최상"
        elif score >= 70:
            interpretation = "매수: 상승 가능성이 있으나 주의 필요"
        elif score >= 60:
            interpretation = "관망: 단기 반등 가능, 신중한 접근 권장"
        elif score >= 40:
            interpretation = "주의: 투자 리스크 높음, 소량/관망 추천"
        else:
            interpretation = "매도: 현재 매수보다는 회피 권장, 하락 위험 높음"

        trend_text = "상승장" if last_close > ma20 else "하락장"
        candle_text = "양봉" if change_pct >= 0 else "음봉"
        vol_text = "폭발!" if vol_ratio >= 3 else "보통"

        st.markdown(f"""
        <div style="background:#111; color:white; padding:20px; border-radius:18px; margin:20px 0;
                    border:2px solid #00ffcc; box-shadow:0 0 20px #00ffcc66;">
            <h3 style="color:#aaa; margin:10px 0;">AI SCORE 해석</h3>
            <p style="font-size:1.3em; margin:5px 0;">점수: {score} / 등급: {grade}</p>
            <p style="font-size:1.2em; margin:5px 0;">해석: {interpretation}</p>
            <p style="font-size:1.2em; margin:5px 0;">추세: {trend_text} | 캔들: {candle_text} | 거래량: {vol_text}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
