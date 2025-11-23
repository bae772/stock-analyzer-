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
        # 실시간 데이터 포함 최근 90일치
        data = yf.download(ticker, period="3mo", interval="1d", progress=False, auto_adjust=True)
        if data.empty or len(data)<5:
            st.warning(f"{ticker}: 데이터 부족")
            continue

        data = data[['Close','High','Low','Volume']].dropna()
        close = data['Close']
        last_close = float(close.iloc[-1])
        last_date = close.index[-1].strftime('%Y-%m-%d')

        # ── 퍼센트 계산 (정확히 1일/1주/3달 전 기준)
        pct_changes = {}
        ref_days = {
            "1일": 1,
            "1주": 5,
            "3달": len(close)-1 if len(close)>=63 else len(close)-1
        }
        for name, days_back in ref_days.items():
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

        # ── 매수/매도 근거 깔끔하게
        if last_rsi<30 and last_macd_diff>0 and last_close<last_lower:
            signal_text = "강력 매수"
        elif last_rsi<40 and last_macd_diff>0:
            signal_text = "매수"
        elif last_rsi>70 or last_close>last_upper:
            signal_text = "매도"
        else:
            signal_text = "관망"

        # ── 평단가 대비 수익률
        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text="평단가 입력 없음"

        # ── UI
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**평단가 대비 수익률:** {profit_text}")
            st.markdown(f"**매수/매도 근거:** {signal_text}")
            st.markdown("**기간별 등락률:**")
            for name, pct in pct_changes.items():
                if pct is not None:
                    st.markdown(f"- {name} 전 대비: {pct}%")

            # ── 차트
            chart_len = 90
            chart_close = close[-chart_len:].reset_index(drop=True)
            chart_ma20 = ma20[-chart_len:].reset_index(drop=True)
            chart_ma50 = ma50[-chart_len:].reset_index(drop=True)
            chart_upper = upper_band[-chart_len:].reset_index(drop=True)
            chart_lower = lower_band[-chart_len:].reset_index(drop=True)
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
        st.error(f"{ticker} 오류 발생: {e}")
