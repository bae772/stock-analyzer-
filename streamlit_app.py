# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        data = data[['Close', 'High', 'Low', 'Volume']].copy()

        # 스칼라 변환
        last_close = float(data['Close'].iloc[-1])
        prev_close = float(data['Close'].iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI 14
        delta = data['Close'].diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - 100 / (1 + rs)

        # 이동평균
        data['MA20'] = data['Close'].rolling(20).mean()
        data['MA50'] = data['Close'].rolling(50).mean()

        # 거래량 비율
        data['VolRatio'] = data['Volume'] / data['Volume'].rolling(20).mean()

        # Target = 다음날 종가
        data['Target'] = data['Close'].shift(-1)

        # 학습용 데이터 준비
        features = ['RSI', 'MA20', 'MA50', 'Volume', 'VolRatio']
        data_model = data.dropna(subset=features + ['Target'])

        if data_model.empty:
            st.warning("모델 학습용 데이터가 충분하지 않습니다. AI 예측은 제공되지 않습니다.")
            ai_score = 50
        else:
            X = data_model[features]
            y = data_model['Target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # 마지막 데이터로 예측
            last_features = data[features].iloc[-1].values.reshape(1, -1)
            pred_next_close = float(model.predict(last_features)[0])
            ai_score = round(50 + (pred_next_close - last_close)/last_close*100)  # 50 기준, 상승폭 반영
            ai_score = int(np.clip(ai_score, 0, 100))

        # 등급
        if ai_score >= 80:
            grade, gcolor = "A (강력매수)", "#00ff00"
        elif ai_score >= 70:
            grade, gcolor = "A (매수)", "#33ff33"
        elif ai_score >= 60:
            grade, gcolor = "B (관망)", "#ffff33"
        elif ai_score >= 40:
            grade, gcolor = "C (주의)", "#ff9933"
        else:
            grade, gcolor = "D (매도)", "#ff3333"

        # ATR 기반 목표가 & 손절가
        tr1 = data['High'] - data['Low']
        tr2 = abs(data['High'] - data['Close'].shift(1))
        tr3 = abs(data['Low']  - data['Close'].shift(1))
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        target_price = round(last_close + atr * 2.5, 2)
        stop_price   = round(last_close - atr * 1.8, 2)
        target_pct   = round((target_price - last_close) / last_close * 100, 1)
        stop_pct     = round((stop_price - last_close) / last_close * 100, 1)

        # ────── 베짱이 계산기 카드 ──────
        st.markdown(f"""
        <div style="background:#000; color:white; padding:30px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:4.5em;">{ticker}</h1>
            <h2 style="margin:10px 0; font-size:3em;">${last_close:.2f}</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.8em; margin:5px;">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>

            <h3 style="color:#aaa; margin:25px 0 5px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if ai_score>=75 else '#ffff00' if ai_score>=60 else '#ff9933'};
                       font-size:7em; margin:0; text-shadow:0 0 30px;">
                {ai_score}
            </h1>

            <h2 style="color:{gcolor}; margin:20px 0;">등급 [{grade}]</h2>

            <div style="display:flex; justify-content:center; gap:40px; margin:20px 0; font-size:1.3em;">
                <div>추세 <span style="color:#33ff33;">{'상승장' if last_close > float(data['MA20'].iloc[-1]) else '하락장'}</span></div>
                <div>캔들 <span style="color:#ffff33;">{'양봉' if change_pct>=0 else '음봉'}</span></div>
            </div>

            <div style="color:#00ffcc; font-size:1.5em; margin:20px 0;">
                거래량 <b>{float(data['VolRatio'].iloc[-1]):.1f}배</b> 
                {'<span style="color:#00ff00;">(폭발!)</span>' if float(data['VolRatio'].iloc[-1]) >= 3 else ''}
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

        # ────── 투자 전략 안내 ──────
        if ai_score >= 80:
            interpretation = "강력매수: 상승 가능성 높음"
            strategy_long = "장기: 적극 매수, 분할 매수 가능"
            strategy_short = "단기: 단기 변동 대비 일부 매수"
        elif ai_score >= 70:
            interpretation = "매수: 상승 가능성 있음"
            strategy_long = "장기: 소량 매수, 추세 확인 후 추가"
            strategy_short = "단기: 단기 관망 후 소량 매수"
        elif ai_score >= 60:
            interpretation = "관망: 신중 접근"
            strategy_long = "장기: 관망 위주"
            strategy_short = "단기: 소량만 매수"
        elif ai_score >= 40:
            interpretation = "주의: 투자 리스크 높음"
            strategy_long = "장기: 관망 또는 소량 매수"
            strategy_short = "단기: 손절가 설정, 관망 추천"
        else:
            interpretation = "매도: 하락 위험 높음"
            strategy_long = "장기: 매수 지양"
            strategy_short = "단기: 손절가 엄격 관리"

        st.markdown(f"""
        <div style="background:#111; color:white; padding:20px; border-radius:18px; margin:20px 0;
                    border:2px solid #00ffcc; box-shadow:0 0 20px #00ffcc66;">
            <h3 style="color:#aaa; margin:10px 0;">AI SCORE 해석 & 투자 전략</h3>
            <p style="font-size:1.3em; margin:5px 0;">점수: {ai_score} / 등급: {grade}</p>
            <p style="font-size:1.2em; margin:5px 0;">해석: {interpretation}</p>
            <p style="font-size:1.2em; margin:5px 0;">장기 투자 전략: {strategy_long}</p>
            <p style="font-size:1.2em; margin:5px 0;">단기 투자 전략: {strategy_short}</p>
            <p style="font-size:1.2em; margin:5px 0; font-weight:bold;">목표가: ${target_price} / 손절가: ${stop_price}</p>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
