# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기 (미국주식 AI 분석 + 투자 전략)")

# ── 사용자 입력 ──
ticker = st.text_input("티커 입력 (예: BMR, TSLA, MARA)", value="BMR").upper()
avg_price = st.number_input("보유 평단가 입력 (없으면 0)", min_value=0.0, step=0.01, value=0.0)

# ── 데이터 다운로드 ──
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
        change_pct = round((last_close - prev_close)/prev_close*100, 2)
        last_date = close.index[-1].strftime('%Y-%m-%d')

        # ── 기본 기술적 지표 ──
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100/(1+rs)
        last_rsi = float(rsi.iloc[-1])

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])

        vol_ma20 = volume.rolling(20).mean()
        last_vol_ma20 = float(vol_ma20.iloc[-1])
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / last_vol_ma20, 2) if last_vol_ma20 > 0 else 1.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        last_macd_diff = float(macd_diff.iloc[-1])

        std20 = close.rolling(20).std()
        upper_band = ma20 + 2*std20
        lower_band = ma20 - 2*std20
        last_upper = float(upper_band.iloc[-1])
        last_lower = float(lower_band.iloc[-1])

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

        # ── 고급 지표 ──
        def stochastic_rsi(close, period=14):
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            rsi_val = 100 - 100/(1 + (gain.rolling(period).mean()/loss.rolling(period).mean()))
            stoch = (rsi_val - rsi_val.rolling(period).min()) / (rsi_val.rolling(period).max() - rsi_val.rolling(period).min())
            return stoch
        data['StochRSI'] = stochastic_rsi(close)

        def cci(high, low, close, period=20):
            tp = (high + low + close)/3
            return (tp - tp.rolling(period).mean()) / (0.015 * tp.rolling(period).std())
        data['CCI'] = cci(high, low, close)

        obv = [0]
        for i in range(1, len(close)):
            obv.append(obv[-1] + (volume.iloc[i] if close.iloc[i]>close.iloc[i-1] else -volume.iloc[i] if close.iloc[i]<close.iloc[i-1] else 0))
        data['OBV'] = obv

        def adx(high, low, close, period=14):
            plus_dm = high.diff()
            minus_dm = low.diff().abs()
            tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
            atr_val = tr.rolling(period).mean()
            plus_di = 100 * (plus_dm.rolling(period).sum()/atr_val)
            minus_di = 100 * (minus_dm.rolling(period).sum()/atr_val)
            dx = 100 * abs(plus_di - minus_di)/(plus_di + minus_di)
            return dx.rolling(period).mean()
        data['ADX'] = adx(high, low, close)

        # ── AI SCORE ──
        score = 50.0
        score += max(0, 30 - last_rsi)*1.4
        score += change_pct*2.0
        score += max(0, vol_ratio-1)*12
        score += 15 if last_close>last_ma20 else -10
        score += 10 if last_close>last_ma50 else -8
        score += 10 if last_macd_diff>0 else -10
        score += 5 if last_close<last_lower else -5
        # 고급 지표 가중치
        score += 5 if data['StochRSI'].iloc[-1]<0.2 else -5
        score = int(np.clip(score,0,100))

        if score>=80:
            grade, gcolor = "A (강력매수)", "#00ff00"
            reason = "RSI/추세/거래량/MACD/스톡래틱RSI 등 긍정적 신호"
        elif score>=70:
            grade, gcolor = "A (매수)", "#33ff33"
            reason = "추세 상승 + 일부 지표 긍정"
        elif score>=60:
            grade, gcolor = "B (관망)", "#ffff33"
            reason = "단기 변동성 존재, 신중 관망"
        elif score>=40:
            grade, gcolor = "C (주의)", "#ff9933"
            reason = "과열/하락 위험, 일부 매수 가능"
        else:
            grade, gcolor = "D (매도)", "#ff3333"
            reason = "과열/하락 신호 다수, 매수 지양"

        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text = "평단가 입력 없음"

        # ── Signal & 전략 ──
        if last_rsi < 30 and last_macd_diff > 0 and last_close < last_lower:
            Buy_Signal, Sell_Signal = True, False
            signal_reason = "강력 매수: RSI 매우 저평가 + MACD 골든크로스 + 볼린저 하단"
        elif last_rsi < 40 and last_macd_diff > 0:
            Buy_Signal, Sell_Signal = True, False
            signal_reason = "매수: RSI 저평가 + MACD 골든크로스"
        elif last_rsi > 70 or last_close > last_upper:
            Buy_Signal, Sell_Signal = False, True
            signal_reason = "매도: RSI 과열 + 볼린저 상단"
        else:
            Buy_Signal = Sell_Signal = False
            signal_reason = "관망: 단기 신호 불확실"

        if Buy_Signal and vol_ratio > 1.5:
            short_strategy = "단기: 변동성 급등 구간 소량 매수"
        elif Buy_Signal:
            short_strategy = "단기: 일반 매수 가능"
        elif Sell_Signal and vol_ratio > 1.5:
            short_strategy = "단기: 과열 구간 일부 매도"
        elif Sell_Signal:
            short_strategy = "단기: 일부 매도 조정"
        else:
            short_strategy = "단기: 관망"

        if last_close > last_ma20 and last_ma20 > last_ma50:
            long_strategy = "장기: 상승추세, 비중 확대"
        elif last_close < last_ma20 and last_ma20 < last_ma50:
            long_strategy = "장기: 하락추세, 신규 매수 지양"
        else:
            long_strategy = "장기: 관망, 추세 확인 필요"

        # ── 카드 출력 ──
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
            <p style="font-size:1.3em; margin:10px 0;">Signal: {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'}</p>
            <p style="font-size:1.3em; margin:10px 0;">Signal Reason: {signal_reason}</p>
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

        # ── 차트 출력 ──
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
