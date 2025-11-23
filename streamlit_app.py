# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime
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

# ── 실시간 시세만
period, interval = "7d", "5m"  # 실시간 시세

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
        last_date = close.index[-1].strftime('%Y-%m-%d %H:%M')

        # ── 1일, 1주, 1개월, 3개월 전 대비 퍼센트 계산
        pct_changes = {}
        periods_back = {"1일":1, "1주":5, "1개월":22, "3달":66}
        for name, days_back in periods_back.items():
            if len(close) > days_back:
                past_close = float(close.iloc[-days_back])
                pct = round((last_close - past_close)/past_close*100, 2)
                pct_changes[name] = pct
            else:
                pct_changes[name] = None

        # ── RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100/(1+rs)
        last_rsi = float(rsi.iloc[-1])

        # ── 이동평균
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        last_ma20 = float(ma20.iloc[-1])
        last_ma50 = float(ma50.iloc[-1])

        # ── 거래량
        vol_ma20 = volume.rolling(20).mean()
        last_vol_ma20 = float(vol_ma20.iloc[-1])
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / last_vol_ma20,2) if last_vol_ma20>0 else 1.0

        # ── MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        last_macd_diff = float(macd_diff.iloc[-1])

        # ── 볼린저밴드
        std20 = close.rolling(20).std()
        upper_band = ma20 + 2*std20
        lower_band = ma20 - 2*std20
        last_upper = float(upper_band.iloc[-1])
        last_lower = float(lower_band.iloc[-1])

        # ── AI SCORE 계산 (1일 기준 변화율 포함)
        base_score = 50.0
        base_score += max(0,30-last_rsi)*1.4*score_weights.get("BUY",1.0)
        change_pct = pct_changes["1일"]
        base_score += change_pct*2.0*score_weights.get("BUY",1.0) if change_pct is not None else 0
        base_score += max(0,vol_ratio-1)*12*score_weights.get("BUY",1.0)
        base_score += 15 if last_close>last_ma20 else -10
        base_score += 10 if last_close>last_ma50 else -8
        base_score += 10 if last_macd_diff>0 else -10
        base_score += 5 if last_close<last_lower else -5
        score = int(np.clip(base_score,0,100))

        if score >= 80:
            grade, reason = "A (강력매수)", "RSI 저평가 + 상승추세 + MACD 골든크로스 등 긍정적 신호"
        elif score >= 70:
            grade, reason = "A (매수)", "추세 상승 + 일부 기술적 지표 긍정적"
        elif score >= 60:
            grade, reason = "B (관망)", "단기 변동성 존재, 신중 관망 필요"
        elif score >= 40:
            grade, reason = "C (주의)", "과열/하락 위험, 일부 매수 가능성만"
        else:
            grade, reason = "D (매도)", "과열/하락 신호 다수, 매수 지양"

        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text="평단가 입력 없음"

        # ── Signal 판단
        if last_rsi<30 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="강력 매수"
        elif last_rsi<40 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="매수"
        elif last_rsi>70 or last_close>last_upper:
            Buy_Signal=False; Sell_Signal=True; signal_reason="매도"
        else:
            Buy_Signal=Sell_Signal=False; signal_reason="관망"

        # ── 아코디언 UI 표시
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**AI Score:** {score} [{grade}]")
            st.markdown(f"**매수/매도 근거:** {reason}")
            st.markdown(f"**평단가 대비 수익률:** {profit_text}")
            st.markdown(f"**Signal:** {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'} ({signal_reason})")
            st.markdown(f"**단기 전략:** {'매수 추천' if Buy_Signal else '매도 추천' if Sell_Signal else '관망'}")
            st.markdown(f"**장기 전략:** {'상승추세' if last_close>last_ma20 else '하락추세 또는 관망'}")

            st.markdown("**1일/1주/1개월/3개월 전 대비 퍼센트:**")
            for name, pct in pct_changes.items():
                if pct is not None:
                    direction = "상승" if pct > 0 else "하락"
                    st.markdown(f"- {name} 전 대비: {abs(pct)}% {direction}")

    except Exception as e:
        st.error(f"{ticker} 오류 발생: {e}")
