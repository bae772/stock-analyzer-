# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="wide")
st.title("베짱이 계산기")

# ── 사용자 입력 ──
tickers_input = st.text_input("티커 입력 (쉼표로 구분, 예: TSLA,AAPL)", value="TSLA")
avg_price_input = st.text_input("보유 평단가 입력 (쉼표로 구분, 없으면 0)", value="0")

tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
avg_prices = [float(x) for x in avg_price_input.split(",")] if avg_price_input else [0]*len(tickers)
if len(avg_prices)<len(tickers):
    avg_prices += [0]*(len(tickers)-len(avg_prices))

# ── 기간 선택 ──
time_options = ["실시간","1일","1주","1개월","3개월","6개월","1년","5년","10년","15년"]
selected_time = st.selectbox("기간 선택", time_options)

period_map = {
    "실시간": ("7d","5m"),
    "1일": ("2d","15m"),
    "1주": ("7d","30m"),
    "1개월": ("1mo","1d"),
    "3개월": ("3mo","1d"),
    "6개월": ("6mo","1d"),
    "1년": ("1y","1d"),
    "5년": ("5y","1d"),
    "10년": ("10y","1d"),
    "15년": ("15y","1d")
}
period, interval = period_map[selected_time]

# ── 기간별 등락률 계산 함수 ──
def get_period_change(adj_close, days_back):
    if len(adj_close) > days_back:
        past = float(adj_close.iloc[-days_back-1])
        last = float(adj_close.iloc[-1])
        return round((last - past)/past*100,2)
    else:
        return None

# ── 각 종목 처리 ──
for idx, ticker in enumerate(tickers):
    avg_price = avg_prices[idx] if idx<len(avg_prices) else 0
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty or len(data)<5:
            st.warning(f"{ticker}: 데이터 부족")
            continue

        data = data.dropna(subset=["Close","High","Low","Volume","Adj Close"])
        close = data["Adj Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

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
        vol_ratio = round(volume.iloc[-1]/vol_ma20.iloc[-1],2) if vol_ma20.iloc[-1]>0 else 1.0

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

        # AI SCORE
        base_score = 50.0
        base_score += max(0,30-last_rsi)*1.4
        base_score += change_pct*2.0
        base_score += max(0,vol_ratio-1)*12
        base_score += 15 if last_close>last_ma20 else -10
        base_score += 10 if last_close>last_ma50 else -8
        base_score += 10 if last_macd_diff>0 else -10
        base_score += 5 if last_close<last_lower else -5
        score = int(np.clip(base_score,0,100))

        # 등급
        if score>=80: grade="A (강력매수)"
        elif score>=70: grade="A (매수)"
        elif score>=60: grade="B (관망)"
        elif score>=40: grade="C (주의)"
        else: grade="D (매도)"

        # Signal
        if last_rsi<30 and last_macd_diff>0 and last_close<last_lower:
            Buy_Signal=True; Sell_Signal=False; signal_reason="강력 매수"
        elif last_rsi<40 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="매수"
        elif last_rsi>70 or last_close>last_upper:
            Buy_Signal=False; Sell_Signal=True; signal_reason="매도"
        else:
            Buy_Signal=Sell_Signal=False; signal_reason="관망"

        # 단기/장기 전략
        short_strategy = "단기: " + ("매수 추천" if Buy_Signal else "매도 추천" if Sell_Signal else "관망")
        long_strategy = "장기: 상승추세" if last_close>last_ma20 else "장기: 하락추세 또는 관망"

        # 기간별 등락률
        period_days_map = {
            "1일":1, "1주":5, "1개월":21, "3개월":63, "6개월":126, "1년":252,
            "5년":252*5, "10년":252*10, "15년":252*15
        }
        days_back = period_days_map.get(selected_time,1)
        period_change = get_period_change(close, days_back)

        # 매수/매도 근거 깔끔하게
        reason_lines = [
            f"RSI: {last_rsi:.2f} ({'저평가' if last_rsi<30 else '과열' if last_rsi>70 else '보통'})",
            f"MACD Diff: {last_macd_diff:.2f} ({'골든크로스' if last_macd_diff>0 else '데드크로스'})",
            f"종가 > MA20: {last_close>last_ma20}",
            f"종가 > MA50: {last_close>last_ma50}",
            f"볼린저 상/하단: {last_upper:.2f}/{last_lower:.2f}, 종가: {last_close:.2f}",
            f"거래량: 평소 대비 {vol_ratio}배"
        ]
        reason_text = " | ".join(reason_lines)

        # UI
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**AI Score:** {score} [{grade}]")
            if period_change is not None:
                st.markdown(f"**{selected_time} 변동률:** {period_change:.2f}%")
            else:
                st.markdown(f"**{selected_time} 변동률:** 데이터 부족")
            st.markdown(f"**매수/매도 근거:** {reason_text}")
            st.markdown(f"**Signal:** {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'} ({signal_reason})")
            st.markdown(f"**단기 전략:** {short_strategy}")
            st.markdown(f"**장기 전략:** {long_strategy}")

        # Plotly 차트
        chart_len = 250
        chart_close = close[-chart_len:].reset_index(drop=True)
        x_axis = list(range(len(chart_close)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=chart_close, mode='lines', name='Adj Close'))
        fig.update_layout(title=f"{ticker} 차트", xaxis_title="기간", yaxis_title="가격", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"{ticker} 오류 발생: {e}")
