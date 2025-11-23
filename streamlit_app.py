# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="wide")
st.title("베짱이 계산기")

# ── 자동 학습 함수 ──
def auto_learn(history_csv="prediction_history.csv"):
    if not os.path.exists(history_csv):
        return {}
    df = pd.read_csv(history_csv)
    df = df.dropna(subset=["예측결과", "AI_SCORE"])
    if df.empty:
        return {}
    results = {}
    for signal_type in ["BUY", "SELL", "HOLD"]:
        sig_df = df[df["Signal"]==signal_type]
        if len(sig_df)==0: continue
        accuracy = len(sig_df[sig_df["예측결과"]=="맞음"]) / len(sig_df)
        results[signal_type] = accuracy
    score_adjust = {}
    score_adjust["BUY"] = results.get("BUY", 0.5) * 1.0
    score_adjust["SELL"] = results.get("SELL", 0.5) * 1.0
    score_adjust["HOLD"] = results.get("HOLD", 0.5) * 1.0
    return score_adjust

history_file = "prediction_history.csv"
score_weights = auto_learn(history_file)

# ── 사용자 입력 ──
tickers_input = st.text_input("티커 입력 (쉼표로 구분, 예: BMR,TSLA,MARA)", value="BMR")
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
def get_change_pct(data, days_back):
    if len(data) > days_back:
        past_close = float(data['Close'].iloc[-days_back-1])
        last_close = float(data['Close'].iloc[-1])
        return round((last_close - past_close) / past_close * 100, 2)
    else:
        return None  # 데이터 부족 시 None 처리

# ── 각 종목 처리 ──
for idx, ticker in enumerate(tickers):
    avg_price = avg_prices[idx] if idx<len(avg_prices) else 0
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if data.empty or len(data)<5:
            st.warning(f"{ticker}: 데이터 부족")
            continue

        data = data[['Close','High','Low','Volume']].dropna()
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
        vol_ratio = round(vol_today / last_vol_ma20,2) if last_vol_ma20>0 else 1.0

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
        tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5,2)
        stop_price = round(last_close - last_atr*1.8,2)

        # AI SCORE
        base_score = 50.0
        base_score += max(0,30-last_rsi)*1.4*score_weights.get("BUY",1.0)
        base_score += change_pct*2.0*score_weights.get("BUY",1.0)
        base_score += max(0,vol_ratio-1)*12*score_weights.get("BUY",1.0)
        base_score += 15 if last_close>last_ma20 else -10
        base_score += 10 if last_close>last_ma50 else -8
        base_score += 10 if last_macd_diff>0 else -10
        base_score += 5 if last_close<last_lower else -5
        score = int(np.clip(base_score,0,100))

        # 기간별 등락률 계산
        period_days_map = {
            "1일": 1,
            "1주": 5,
            "1개월": 21,
            "3개월": 63,
            "6개월": 126,
            "1년": 252,
            "5년": 252*5,
            "10년": 252*10,
            "15년": 252*15
        }
        days_back = period_days_map.get(selected_time,1)
        period_change = get_change_pct(data, days_back)

        # ── 아코디언 UI ──
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**기간 변동률:** {period_change:.2f}%" if period_change is not None else "**기간 변동률:** 데이터 부족")
            st.markdown(f"**RSI:** {last_rsi:.2f}")
            st.markdown(f"**MACD Diff:** {last_macd_diff:.2f}")
            st.markdown(f"**MA20:** {last_ma20:.2f}, MA50: {last_ma50:.2f}")
            st.markdown(f"**볼린저밴드:** 상단 {last_upper:.2f}, 하단 {last_lower:.2f}")
            st.markdown(f"**거래량 비율:** {vol_ratio}배")
            st.markdown(f"**AI Score:** {score}")

        # Plotly 차트
        chart_len = 250
        chart_close = close[-chart_len:].reset_index(drop=True)
        x_axis = list(range(len(chart_close)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=chart_close, mode='lines', name='Close'))
        fig.update_layout(title=f"{ticker} 차트", xaxis_title="기간", yaxis_title="가격", template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"{ticker} 오류 발생: {e}")
