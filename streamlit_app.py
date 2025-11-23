# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기 (미국주식 AI 분석 + 투자 전략)")

# 사용자 입력
ticker = st.text_input("티커 입력 (예: BMR, TSLA, MARA)", value="BMR").upper()
avg_price = st.number_input("보유 평단가 입력 (없으면 0)", min_value=0.0, step=0.01, value=0.0)

if ticker:
    try:
        # 데이터 다운로드
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        data = data[['Close', 'High', 'Low', 'Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close)/prev_close*100,2)
        last_date = close.index[-1].strftime('%Y-%m-%d')

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100/(1+rs)
        last_rsi = float(rsi.iloc[-1])

        # 이동평균
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])

        # 거래량
        vol_ma20 = volume.rolling(20).mean()
        last_vol_ma20 = float(vol_ma20.iloc[-1])
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / last_vol_ma20, 2) if last_vol_ma20 > 0 else 1.0

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        last_macd_diff = float(macd_diff.iloc[-1])

        # 볼린저밴드
        std20 = close.rolling(20).std()
        upper_band = ma20 + 2*std20
        lower_band = ma20 - 2*std20
        last_upper = float(upper_band.iloc[-1])
        last_lower = float(lower_band.iloc[-1])

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5, 2)
        stop_price = round(last_close - last_atr*1.8, 2)
        target_pct = round((target_price - last_close)/last_close*100,1)
        stop_pct = round((stop_price - last_close)/last_close*100,1)

        # AI SCORE
        score = 50.0
        score += max(0, 30 - last_rsi)*1.4
        score += change_pct*2.0
        score += max(0, vol_ratio-1)*12
        score += 15 if last_close>last_ma20 else -10
        score += 10 if last_close>last_ma50 else -8
        score += 10 if last_macd_diff>0 else -10
        score += 5 if last_close<last_lower else -5
        score = int(np.clip(score,0,100))

        # 등급 + 근거
        if score>=80:
            grade, gcolor = "A (강력매수)", "#00ff00"
            reason = "RSI 저평가 + 상승추세 + MACD 골든크로스 등 긍정적 신호"
        elif score>=70:
            grade, gcolor = "A (매수)", "#33ff33"
            reason = "추세 상승 + 일부 기술적 지표 긍정적"
        elif score>=60:
            grade, gcolor = "B (관망)", "#ffff33"
            reason = "단기 변동성 존재, 신중 관망 필요"
        elif score>=40:
            grade, gcolor = "C (주의)", "#ff9933"
            reason = "과열/하락 위험, 일부 매수 가능성만"
        else:
            grade, gcolor = "D (매도)", "#ff3333"
            reason = "과열/하락 신호 다수, 매수 지양"

        # 평단가 대비 수익률
        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text = "평단가 입력 없음"

        # ── 단기/장기 투자 전략 추천 ──
        if last_rsi < 40 and last_macd_diff > 0 and change_pct >= -2:
            short_strategy = "단기: 소량 매수 가능 (RSI 저평가 + MACD 골든크로스)"
        elif last_rsi > 70 or last_close > last_upper:
            short_strategy = "단기: 관망 또는 일부 매도 (과열 구간)"
        else:
            short_strategy = "단기: 관망"

        if last_close > last_ma20 and last_ma20 > last_ma50:
            long_strategy = "장기: 상승추세, 비중 확대 가능"
        elif last_close < last_ma20 and last_ma20 < last_ma50:
            long_strategy = "장기: 하락 추세, 신규 매수 지양"
        else:
            long_strategy = "장기: 관망"

        # 카드 출력
        st.markdown(f"""
        <div style="background:#000; color:white; padding:30px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:4.5em;">{ticker}</h1>
            <h2 style="margin:10px 0; font-size:3em;">${last_close:.2f} ({last_date})</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.8em; margin:5px;">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>
            <h3 style="color:#aaa; margin:25px 0 5px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:7em; margin:0; text-shadow:0 0 30px;">
                {score}
            </h1>
            <h2 style="color:{gcolor}; margin:20px 0;">등급 [{grade}]</h2>
            <p style="font-size:1.3em; margin:10px 0;">매수/매도 근거: {reason}</p>
            <p style="font-size:1.3em; margin:10px 0;">평단가 대비 수익률: {profit_text}</p>
            <p style="font-size:1.3em; margin:10px 0;">단기 전략: {short_strategy}</p>
            <p style="font-size:1.3em; margin:10px 0;">장기 전략: {long_strategy}</p>
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

        # Plotly 차트
        chart_len = 250
        chart_close = close[-chart_len:].dropna().reset_index(drop=True)
        chart_ma20 = ma20[-chart_len:].dropna().reset_index(drop=True)
        chart_ma50 = ma50[-chart_len:].dropna().reset_index(drop=True)
        chart_upper = upper_band[-chart_len:].dropna().reset_index(drop=True)
        chart_lower = lower_band[-chart_len:].dropna().reset_index(drop=True)
        x_axis = list(range(len(chart_close)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=chart_close, mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=x_axis, y=chart_ma20, mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=x_axis, y=chart_ma50, mode='lines', name='MA50'))
        fig.add_trace(go.Scatter(x=x_axis, y=chart_upper, mode='lines', name='Upper BB', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=x_axis, y=chart_lower, mode='lines', name='Lower BB', line=dict(dash='dot')))
        fig.update_layout(title=f"{ticker} 차트", xaxis_title="기간", yaxis_title="가격", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"오류 발생: {e}")
