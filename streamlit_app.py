# streamlit_ml_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os

st.set_page_config(page_title="학습형 AI 스코어 카드", layout="centered")
st.title("학습형 AI 스코어 카드 (미국주식)")

ticker = st.text_input("티커 입력 (예: BMR, TSLA, MARA)", value="BMR").upper()

MODEL_FILE = f"{ticker}_ml_model.pkl"

if ticker:
    try:
        # 1년치 데이터
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 60:
            st.error("데이터 부족 또는 티커를 찾을 수 없습니다.")
            st.stop()
        
        # 필요한 열
        data = data[['Close', 'High', 'Low', 'Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        # ────── 특징(feature) 생성 ──────
        df = pd.DataFrame()
        df['rsi14'] = 100 - 100 / (1 + close.diff().clip(lower=0).rolling(14).mean() / 
                                    (-close.diff().clip(upper=0).rolling(14).mean()))
        df['ma20_diff'] = close / close.rolling(20).mean() - 1
        df['ma50_diff'] = close / close.rolling(50).mean() - 1
        df['vol_ratio'] = volume / volume.rolling(20).mean()
        df['target'] = (close.shift(-1) > close).astype(int)  # 다음 날 상승 여부
        df = df.dropna()

        X = df[['rsi14','ma20_diff','ma50_diff','vol_ratio']]
        y = df['target']

        # ────── 모델 학습 또는 불러오기 ──────
        if os.path.exists(MODEL_FILE):
            model = joblib.load(MODEL_FILE)
        else:
            model = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42)
            model.fit(X, y)
            joblib.dump(model, MODEL_FILE)

        # 최신 데이터로 예측
        latest_features = X.iloc[-1].values.reshape(1, -1)
        prob = model.predict_proba(latest_features)[0][1]  # 상승 확률
        score = int(prob * 100)

        # 등급 설정
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

        st.markdown(f"""
        <div style="background:#000; color:white; padding:25px; border-radius:18px; text-align:center; 
                    border:3px solid #00ffcc; box-shadow:0 0 25px #00ffcc88;">
            <h1 style="color:#00ffcc; margin:0; font-size:4em;">{ticker}</h1>
            <h3 style="color:#aaa; margin:15px 0 5px;">학습형 AI SCORE</h3>
            <h1 style="color:{gcolor}; font-size:6em; margin:0; text-shadow:0 0 20px;">
                {score}
            </h1>
            <h2 style="color:{gcolor}; margin:20px 0;">등급 [{grade}]</h2>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
