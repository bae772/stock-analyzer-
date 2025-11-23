# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="wide")
st.title("베짱이 계산기")

# ── 사용자 입력 ──
tickers_input = st.text_input("티커 입력 (쉼표로 구분, 예: BMR,TSLA,MARA)", value="BMR")
avg_price_input = st.text_input("보유 평단가 입력 (쉼표로 구분, 없으면 0)", value="0")

tickers = [x.strip().upper() for x in tickers_input.split(",") if x.strip()]
avg_prices = [float(x) for x in avg_price_input.split(",")] if avg_price_input else [0]*len(tickers)
if len(avg_prices)<len(tickers):
    avg_prices += [0]*(len(tickers)-len(avg_prices))

# ── 각 종목 처리 ──
for idx, ticker in enumerate(tickers):
    avg_price = avg_prices[idx] if idx<len(avg_prices) else 0
    try:
        # 6개월 전까지 데이터 확보
        end = datetime.today()
        start = end - timedelta(days=90)  # 3개월 전까지
        data = yf.download(ticker, start=start, end=end+timedelta(days=1), progress=False, auto_adjust=True)

        if data.empty or len(data)<2:
            st.warning(f"{ticker}: 데이터 부족")
            continue

        close = data['Close']
        last_close = float(close.iloc[-1])
        last_date = close.index[-1].strftime('%Y-%m-%d')

        # ── 정확한 퍼센트 계산 (1일, 1주, 3개월)
        pct_changes = {}
        periods = {"1일":1, "1주":5, "3달":63}  # 대략 영업일 기준
        for name, days_back in periods.items():
            if len(close) > days_back:
                past_close = float(close.iloc[-days_back])
                pct = round((last_close - past_close)/past_close*100,2)
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
        volume = data['Volume']
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

        # ── 매수/매도 근거 (깔끔하게)
        reason_lines = []
        reason_lines.append(f"RSI: {last_rsi:.2f} ({'저평가' if last_rsi<30 else '중립' if last_rsi<70 else '과매수'})")
        reason_lines.append(f"MACD diff: {last_macd_diff:.2f} ({'골든크로스' if last_macd_diff>0 else '데드크로스'})")
        reason_lines.append(f"종가 > MA20: {last_close>last_ma20}")
        reason_lines.append(f"종가 > MA50: {last_close>last_ma50}")
        reason_lines.append(f"볼린저 상단/하단: {last_upper:.2f}/{last_lower:.2f}, 종가: {last_close:.2f}")
        reason_lines.append(f"거래량: 평소 대비 {vol_ratio}배")

        reason_text = "\n".join(reason_lines)

        # ── 평단가 대비 수익률
        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text="평단가 입력 없음"

        # ── Signal
        if last_rsi<30 and last_macd_diff>0 and last_close<last_lower:
            Buy_Signal=True; Sell_Signal=False; signal_reason="강력 매수"
        elif last_rsi<40 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="매수"
        elif last_rsi>70 or last_close>last_upper:
            Buy_Signal=False; Sell_Signal=True; signal_reason="매도"
        else:
            Buy_Signal=Sell_Signal=False; signal_reason="관망"

        # ── 아코디언 UI ──
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**실시간 종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**평단가 대비 수익률:** {profit_text}")
            st.markdown(f"**Signal:** {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'} ({signal_reason})")
            st.markdown(f"**매수/매도 근거:**\n{reason_text}")

            st.markdown("**기간별 등락률:**")
            for p_name, pct in pct_changes.items():
                if pct is not None:
                    st.markdown(f"- {p_name} 전 대비: {pct}%")

    except Exception as e:
        st.error(f"{ticker} 오류 발생: {e}")
