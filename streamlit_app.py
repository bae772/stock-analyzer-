# streamlit_app.py  ← 이 파일명 그대로
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(page_title="키움 AI 카드", layout="centered")

# 코드 안 보이게 + UI 깔끔하게
st.markdown("<style>.css-1d391kg, pre, code, .stAlert {display:none !important;} .block-container {padding-top:2rem;}</style>", unsafe_allow_html=True)

st.title("키움증권 스타일 AI 분석 카드")

ticker = st.text_input("티커 입력", "BMR", label_visibility="collapsed").upper()

if ticker:
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if len(df) < 50:
            st.error("데이터 부족")
            st.stop()

        df = df[['Close','High','Low','Volume']].dropna()
        c, h, l, v = df['Close'], df['High'], df['Low'], df['Volume']

        # 핵심 지표
        price     = float(c.iloc[-1])
        prev      = float(c.iloc[-2])
        chg       = round((price-prev)/prev*100, 2)
        rsi       = float(100 - 100/(1 + (c.diff().clip(lower=0).rolling(14).mean() / (-c.diff().clip(upper=0).rolling(14).mean())).iloc[-1]))
        vol_ratio = round(v.iloc[-1] / v.rolling(20).mean().iloc[-1], 2)
        ma20      = c.rolling(20).mean().iloc[-1]
        ma50      = c.rolling(50).mean().iloc[-1]

        # AI 스코어
        score = 50
        score += max(0, 30-rsi) * 1.4
        score += chg * 2
        score += max(0, vol_ratio-1) * 12
        score += 15 if price > ma20 else -10
        score += 10 if price > ma50 else -8
        score = int(np.clip(score, 0, 100))

        # 등급
        grades = [(80,"A (강력매수)","#00ff00"), (70,"A (매수)","#33ff33"), (60,"B (관망)","#ffff33"),
                  (40,"C (주의)","#ff9933"), (0,"D (매도)","#ff3333")]
        for s, g, col in grades:
            if score >= s:
                grade, gcolor = g, col
                break

        # ATR 목표가·손절가
        tr = pd.concat([h-l, abs(h-c.shift()), abs(l-c.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        target = round(price + atr*2.5, 2)
        stop   = round(price - atr*1.8, 2)
        tpct   = round((target-price)/price*100, 1)
        spct   = round((stop-price)/price*100, 1)

        # 키움 카드
        st.markdown(f"""
        <div style="background:#000;color:#fff;padding:40px;border-radius:25px;text-align:center;
                    border:3px solid #00ffcc;box-shadow:0 0 40px #00ffccaa;margin:20px 0;">
            <h1 style="color:#00ffcc;margin:0;font-size:5em">{ticker}</h1>
            <h2 style="margin:15px 0;font-size:3.5em">${price:.2f}</h2>
            <p style="color:{'#33ff33' if chg>=0 else '#ff3333'};font-size:2.3em">
                {'+' if chg>=0 else ''}{chg}%
            </p>
            <h3 style="color:#aaa;margin:35px 0 10px">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:8em;margin:0">{score}</h1>
            <h2 style="color:{gcolor};margin:30px 0">등급 [{grade}]</h2>
            <div style="color:#00ffcc;font-size:1.8em;margin:25px 0">
                거래량 <b>{vol_ratio:.1f}배</b> {'<span style="color:#00ff00">(폭발!)</span>' if vol_ratio>=3 else ''}
            </div>
            <div style="display:flex;justify-content:center;gap:50px;margin:40px 0">
                <div style="background:#002200;padding:25px;border-radius:18px;min-width:170px">
                    <div style="color:#00ffaa">TARGET</div>
                    <h3 style="color:#00ff00;margin:10px 0 0">${target}<br><small>+{tpct}%</small></h3>
                </div>
                <div style="background:#220000;padding:25px;border-radius:18px;min-width:170px">
                    <div style="color:#ff8888">STOP LOSS</div>
                    <h3 style="color:#ff3333;margin:10px 0 0">${stop}<br><small>{spct}%</small></h3>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 차트
        st.markdown("### 주가·이동평균선·거래량 차트")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=c, name="종가", line=dict(color="#00ffcc", width=3)))
        fig.add_trace(go.Scatter(x=df.index, y=c.rolling(20).mean(), name="20일선", line=dict(color="#33ff33")))
        fig.add_trace(go.Scatter(x=df.index, y=c.rolling(50).mean(), name="50일선", line=dict(color="#ff9933")))
        colors = ['#00ff88' if c.iloc[i] >= c.iloc[i-1] else '#ff4444' for i in range(len(c))]
        fig.add_trace(go.Bar(x=df.index, y=v, name="거래량", marker_color=colors, opacity=0.3, yaxis="y2"))

        fig.update_layout(template="plotly_dark", plot_bgcolor="#000", paper_bgcolor="#000", height=580,
                          legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
                          yaxis=dict(title="가격", showgrid=False),
                          yaxis2=dict(title="거래량", overlaying="y", side="right", showgrid=False),
                          xaxis=dict(showgrid=False), hovermode="x unified",
                          margin=dict(l=20,r=20,t=60,b=20))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.caption(f"업데이트 {datetime.now().strftime('%m-%d %H:%M')}")

    except Exception as e:
        st.error("데이터를 불러올 수 없습니다.")
