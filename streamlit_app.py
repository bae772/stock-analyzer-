import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import base64

st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")
st.title("키움증권 스타일 AI 분석 카드")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        # 필요한 열 사용 + NaN 제거
        data = data[['Close', 'High', 'Low', 'Volume']].dropna()

        close  = data['Close']
        high   = data['High']
        low    = data['Low']
        volume = data['Volume']

        # 스칼라 변환
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        last_rsi = float(rsi.iloc[-1])

        # 거래량 폭발 비율
        vol_ma20 = float(volume.rolling(20).mean().iloc[-1])
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / vol_ma20, 2) if vol_ma20 > 0 else 1.0

        # 이동평균
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])

        # AI SCORE
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

        # ATR for target/stop
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])

        target_price = round(last_close + atr * 2.5, 2)
        stop_price   = round(last_close - atr * 1.8, 2)
        target_pct   = round((target_price - last_close) / last_close * 100, 1)
        stop_pct     = round((stop_price - last_close) / last_close * 100, 1)

        # ---- 스파크라인(미니 차트) 생성 ----
        spark = close.tail(60)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=spark.index,
            y=spark.values,
            mode="lines",
            line=dict(width=2),
            hoverinfo='skip'
        ))

        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=85
        )

        img_bytes = fig.to_image(format="png")
        encoded = base64.b64encode(img_bytes).decode()
        sparkline_html = f'<img src="data:image/png;base64,{encoded}" style="width:100%; opacity:0.95;" />'

        # ---- 카드 UI 출력 ----
        card_html = f"""
        <div style="
            width: 100%;
            padding: 25px;
            background: #000000;
            border-radius: 22px;
            border: 3px solid #00ffe5;
            box-shadow: 0 0 35px #00ffe588;
            text-align: center;
            font-family: 'Segoe UI', sans-serif;
        ">
            
            <div style="font-size: 4em; color:#00faff; font-weight:700; margin-bottom:10px;">
                {ticker}
            </div>

            <div style="font-size: 2.4em; color: white; margin-top:-5px;">
                ${last_close:.2f}
            </div>

            <div style="font-size: 1.6em; margin-top:5px; color:{'#33ff33' if change_pct>=0 else '#ff4444'};">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </div>

            <div style="margin: 20px 0 10px;">
                {sparkline_html}
            </div>

            <div style="color:#aaaaaa; margin-top:20px; font-size:1.3em;">AI SCORE</div>
            <div style="font-size:6em; font-weight:800; 
                        color:{'#00ff00' if score>=75 else '#ffff33' if score>=60 else '#ff9933'};
                        text-shadow:0 0 25px;">
                {score}
            </div>

            <div style="font-size:2em; margin-top:10px; color:{gcolor};">
                등급 [{grade}]
            </div>

            <div style="margin-top:20px; font-size:1.5em; color:#00ffee;">
                거래량 {vol_ratio:.1f}배 
                {'<span style="color:#00ff88;">(폭발!)</span>' if vol_ratio >= 3 else ''}
            </div>

            <div style="display:flex; justify-content:center; gap:35px; margin-top:30px;">

                <div style="background:#002a1f; padding:18px 25px; border-radius:15px;">
                    <div style="color:#00ffcc;">TARGET</div>
                    <div style="color:#00ff88; font-size:1.7em; font-weight:700;">
                        ${target_price} <br>
                        <span style="font-size:0.7em;">+{target_pct}%</span>
                    </div>
                </div>

                <div style="background:#2a0000; padding:18px 25px; border-radius:15px;">
                    <div style="color:#ff7777;">STOP LOSS</div>
                    <div style="color:#ff4444; font-size:1.7em; font-weight:700;">
                        ${stop_price} <br>
                        <span style="font-size:0.7em;">{stop_pct}%</span>
                    </div>
                </div>

            </div>

        </div>
        """

        st.markdown(card_html, unsafe_allow_html=True)
        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
