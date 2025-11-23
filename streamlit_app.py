# ───────────────────────────────────────────────
#   베짱이 계산기 (Energy + Pattern + AI Reason)
# ───────────────────────────────────────────────
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="wide")
st.title("베짱이 계산기")


# ==================================================
# 자동 학습 기존 그대로
# ==================================================
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
    score_adjust = {
        "BUY": results.get("BUY", 0.5),
        "SELL": results.get("SELL", 0.5),
        "HOLD": results.get("HOLD", 0.5)
    }
    return score_adjust


history_file = "prediction_history.csv"
score_weights = auto_learn(history_file)



# ==================================================
# 캔들 패턴 분석 함수  (추가)
# ==================================================
def detect_candle_patterns(df):
    pat_score = 0
    pattern_list = []

    o = df['Open']
    h = df['High']
    l = df['Low']
    c = df['Close']

    # 최근 3일만 사용
    o1, h1, l1, c1 = o.iloc[-1], h.iloc[-1], l.iloc[-1], c.iloc[-1]
    o2, h2, l2, c2 = o.iloc[-2], h.iloc[-2], l.iloc[-2], c.iloc[-2]
    o3, h3, l3, c3 = o.iloc[-3], h.iloc[-3], l.iloc[-3], c.iloc[-3]

    body = abs(c1 - o1)
    candle_range = h1 - l1

    # Doji
    if body <= candle_range * 0.1:
        pattern_list.append("Doji (변곡 가능)")
        pat_score += 5

    # Hammer
    if (c1 > o1) and ((o1 - l1) > body * 2) and ((h1 - c1) < body):
        pattern_list.append("Hammer (반등 신호)")
        pat_score += 10

    # Inverted Hammer
    if (c1 > o1) and ((h1 - o1) > body * 2) and ((o1 - l1) < body):
        pattern_list.append("Inverted Hammer (반등 가능)")
        pat_score += 7

    # Bullish Engulfing
    if (c1 > o1) and (o1 < c2) and (c1 > o2):
        pattern_list.append("Bullish Engulfing (강한 상승 신호)")
        pat_score += 15

    # Bearish Engulfing
    if (c1 < o1) and (o1 > c2) and (c1 < o2):
        pattern_list.append("Bearish Engulfing (하락 신호)")
        pat_score -= 15

    # Morning Star (3일 패턴)
    if (c3 > o3) and (c2 < o2) and (c1 > o1) and (c1 > (o3 + c3) / 2):
        pattern_list.append("Morning Star (강한 반등)")
        pat_score += 20

    # Evening Star (3일 패턴)
    if (c3 < o3) and (c2 > o2) and (c1 < o1) and (c1 < (o3 + c3) / 2):
        pattern_list.append("Evening Star (하락 반전)")
        pat_score -= 20

    return pat_score, pattern_list



# ==================================================
# 에너지 지표 계산 함수 (추가)
# ==================================================
def compute_energy(df):
    close = df['Close']
    volume = df['Volume']

    # Price Momentum Energy
    pm = abs(close.diff()).rolling(14).mean()
    price_energy = np.clip((pm.iloc[-1] / close.iloc[-1]) * 300, 0, 100)

    # Volatility Energy (ATR 기반)
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    vol_energy = np.clip((atr.iloc[-1] / close.iloc[-1]) * 300, 0, 100)

    # Volume Energy
    vol_ma20 = volume.rolling(20).mean()
    vol_energy_score = np.clip((volume.iloc[-1] / vol_ma20.iloc[-1]) * 40, 0, 100)

    # 종합 에너지 점수
    energy_score = int((price_energy * 0.4) + (vol_energy * 0.4) + (vol_energy_score * 0.2))

    return energy_score, price_energy, vol_energy, vol_energy_score



# ==================================================
# 사용자 입력
# ==================================================
tickers_input = st.text_input("티커 입력 (쉼표로 구분)", value="BMR")
avg_price_input = st.text_input("평단가 입력 (없으면 0)", value="0")

tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
avg_prices = [float(x) for x in avg_price_input.split(",")] if avg_price_input else [0]*len(tickers)
if len(avg_prices)<len(tickers):
    avg_prices += [0]*(len(tickers)-len(avg_prices))



# ==================================================
# 기간 선택
# ==================================================
time_options = ["실시간","1일","1주","3달","6달","1년","5년","10년","15년"]
selected_time = st.selectbox("기간 선택", time_options)

period_map = {
    "실시간": ("7d","5m"), 
    "1일": ("2d","15m"), 
    "1주": ("7d","30m"),
    "3달": ("3mo","1d"),
    "6달": ("6mo","1d"),
    "1년": ("1y","1d"),
    "5년": ("5y","1d"),
    "10년": ("10y","1d"),
    "15년": ("15y","1d")
}
period, interval = period_map[selected_time]



# ==================================================
# 종목 루프
# ==================================================
for idx, ticker in enumerate(tickers):

    avg_price = avg_prices[idx]

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df.empty: 
            st.warning(f"{ticker}: 데이터 없음")
            continue

        df["Open"] = df["Close"].shift(1).fillna(df["Close"])  # 단순 오프닝 보정


        # ===== 기존 계산 (RSI, MA, MACD, 볼밴, ATR 동일) =====
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rsi = 100 - 100/(1 + (gain.rolling(14).mean() / loss.rolling(14).mean()))
        last_rsi = float(rsi.iloc[-1])

        # MA
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        macd_diff = macd - signal
        last_macd_diff = float(macd_diff.iloc[-1])

        # 볼린저밴드
        std20 = close.rolling(20).std()
        upper_band = ma20 + 2*std20
        lower_band = ma20 - 2*std20

        # ATR
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5, 2)
        stop_price   = round(last_close - last_atr*1.8, 2)



        # ==================================================
        # 에너지 지표 + 패턴 분석 (추가)
        # ==================================================
        energy_score, price_e, vola_e, vol_e = compute_energy(df)
        pattern_score, pattern_list = detect_candle_patterns(df)



        # ==================================================
        # AI Score 통합 (기존 + 추가)
        # ==================================================
        old_score = 50
        old_score += max(0,30-last_rsi)*1.2
        old_score += change_pct * 2
        old_score += 10 if last_macd_diff>0 else -10
        old_score += 10 if last_close>ma20.iloc[-1] else -8
        old_score += 10 if last_close>ma50.iloc[-1] else -8

        # 추가 반영
        final_ai_score = int(np.clip(old_score*0.6 + energy_score*0.25 + pattern_score*0.15, 0, 100))



        # ==================================================
        # AI 설명 생성
        # ==================================================
        explain = f"""
### AI 분석 요약  
- 에너지 지표: {energy_score}점  
  - Price Energy: {price_e:.1f}  
  - Volatility Energy: {vola_e:.1f}  
  - Volume Energy: {vol_e:.1f}  
- 패턴 점수: {pattern_score}  
- 감지된 패턴: {', '.join(pattern_list) if pattern_list else '없음'}  
- 목표가: {target_price}  
- 손절가: {stop_price}  

### 매수/매도 판단  
AI Score: **{final_ai_score}점**  
"""

        if final_ai_score >= 80:
            final_signal = "강력 매수"
        elif final_ai_score >= 70:
            final_signal = "매수"
        elif final_ai_score >= 55:
            final_signal = "관망"
        elif final_ai_score >= 40:
            final_signal = "주의 / 매도 경계"
        else:
            final_signal = "매도"

        explain += f"→ 최종 판단: **{final_signal}**"



        # ==================================================
        # UI 출력
        # ==================================================
        with st.expander(f"{ticker} 분석 보기", expanded=True):

            st.markdown(f"**AI Score:** {final_ai_score} / 100")
            st.markdown(f"**판단:** {final_signal}")
            st.markdown(explain)

            # 차트
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=close
