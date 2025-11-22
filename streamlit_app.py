# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기 (미국주식)")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        data = data[['Close', 'High', 'Low', 'Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        last_rsi = float(rsi.iloc[-1])

        vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 1.0

        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])

        # AI 스코어 계산
        score = 50.0
        score += max(0, 30 - last_rsi) * 1.4
        score += change_pct * 2.0
        score += max(0, vol_ratio - 1) * 12
        score += 15 if last_close > ma20 else -10
        score += 10 if last_close > ma50 else -8
        score = int(np.clip(score, 0, 100))

        if score >= 80:
            grade, gcolor = "A (강력매수)", "#00ff00"
            interpretation = "강력매수: 상승 가능성이 높음"
            strategy = "적극적 매수, 단기 변동 대비 분할 매수 가능"
        elif score >= 70:
            grade, gcolor = "A (매수)", "#33ff33"
            interpretation = "매수: 상승 가능성이 있으나 주의 필요"
            strategy = "일부 매수, 추세 확인 후 분할 매수"
        elif score >= 60:
            grade, gcolor = "B (관망)", "#ffff33"
            interpretation = "관망: 단기 반등 가능, 신중 접근"
            strategy = "관망 위주, 변동성 큰 구간 소량만"
        elif score >= 40:
            grade, gcolor = "C (주의)", "#ff9933"
            interpretation = "주의: 투자 리스크 높음"
            strategy = "매수보다는 관망, 손절가 설정"
        else:
            grade, gcolor = "D (매도)", "#ff3333"
            interpretation = "매도: 회피 권장, 하락 위험 높음"
            strategy = "매수 지양, 이미 보유 시 손절 관리"

        # ATR 기반 목표/손절
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        target_price = round(last_close + atr * 2.5, 2)
        stop_price = round(last_close - atr * 1.8, 2)
        target_pct = round((target_price - last_close) / last_close * 100, 1)
        stop_pct = round((stop_price - last_close) / last_close * 100, 1)

        # ────── 자동 학습 ──────
        data['RSI'] = rsi
        data['MA20'] = close.rolling(20).mean()
        data['MA50'] = close.rolling(50).mean()
        data['Target'] = close.shift(-1)
        features = ['RSI','MA20','MA50','Volume']
        data_model = data.dropna(subset=features+['Target'])
        X = data_model[features]
        y = data_model['Target']

        model_file = f"{ticker}_model.pkl"
        if os.path.exists(model_file):
            model = joblib.load(model_file)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        joblib.dump(model, model_file)

        next_day_features = X.iloc[-1].values.reshape(1,-1)
        predicted_price = model.predict(next_day_features)[0]

        # ────── 카드 출력 ──────
        st.markdown(f"""
        <div style="background:#000; color:white; padding:25px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:4.5em;">{ticker}</h1>
            <h2 style="margin:10px 0; font-size:3em;">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.6em;">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>
            <h3 style="color:#aaa; margin:15px 0 5px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:7em; margin:0; text-shadow:0 0 30px;">{score}</h1>
            <h2 style="color:{gcolor}; margin:15px 0;">등급 [{grade}]</h2>
            <p style="font-size:1.2em; margin:5px 0;">해석: {interpretation}</p>
            <p style="font-size:1.2em; margin:5px 0;">추천 전략: {strategy}</p>
            <p style="font-size:1.2em; margin:5px 0;">TARGET: ${target_price} (+{target_pct}%) | STOP LOSS: ${stop_price} ({stop_pct}%)</p>
            <p style="font-size:1.2em; margin:5px 0; font-weight:bold;">자동 학습 예측: 내일 종가 ≈ ${predicted_price:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
