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

# ── 기간 선택 ──
time_options = ["실시간","1일","1주","1달","3달","6달","1년","5년","10년","15년"]
selected_time = st.selectbox("기간 선택", time_options)

period_map = {
    "실시간": ("7d","5m"), 
    "1일": ("2d","15m"), 
    "1주": ("7d","30m"),
    "1달": ("1mo","1d"),
    "3달": ("3mo","1d"),
    "6달": ("6mo","1d"),
    "1년": ("1y","1d"),
    "5년": ("5y","1d"),
    "10년": ("10y","1d"),
    "15년": ("15y","1d")
}
period, interval = period_map[selected_time]

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

        # ── 기술적 지표 계산 ──
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
        vol_ratio = round(vol_today / last_vol_ma20,2) if last_vol_ma20>0 else 1.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal_line
        last_macd_diff = float(macd_diff.iloc[-1])

        std20 = close.rolling(20).std()
        upper_band = ma20 + 2*std20
        lower_band = ma20 - 2*std20
        last_upper = float(upper_band.iloc[-1])
        last_lower = float(lower_band.iloc[-1])

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5,2)
        stop_price = round(last_close - last_atr*1.8,2)

        # ── AI SCORE 계산 ──
        base_score = 50.0
        base_score += max(0,30-last_rsi)*1.4*score_weights.get("BUY",1.0)
        base_score += change_pct*2.0*score_weights.get("BUY",1.0)
        base_score += max(0,vol_ratio-1)*12*score_weights.get("BUY",1.0)
        base_score += 15 if last_close>last_ma20 else -10
        base_score += 10 if last_close>last_ma50 else -8
        base_score += 10 if last_macd_diff>0 else -10
        base_score += 5 if last_close<last_lower else -5
        score = int(np.clip(base_score,0,100))

        # ── 등급 및 설명 ──
        if score>=80:
            grade,gcolor="A (강력매수)","#00ff00"
            reason="RSI 저평가 + 상승추세 + MACD 골든크로스 등 긍정적 신호"
        elif score>=70:
            grade,gcolor="A (매수)","#33ff33"
            reason="추세 상승 + 일부 기술적 지표 긍정적"
        elif score>=60:
            grade,gcolor="B (관망)","#ffff33"
            reason="단기 변동성 존재, 신중 관망 필요"
        elif score>=40:
            grade,gcolor="C (주의)","#ff9933"
            reason="과열/하락 위험, 일부 매수 가능성만"
        else:
            grade,gcolor="D (매도)","#ff3333"
            reason="과열/하락 신호 다수, 매수 지양"

        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
        else:
            profit_text="평단가 입력 없음"

        # ── Signal 계산 ──
        if last_rsi<30 and last_macd_diff>0 and last_close<last_lower:
            Buy_Signal=True; Sell_Signal=False; signal_reason="강력 매수"
        elif last_rsi<40 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="매수"
        elif last_rsi>70 or last_close>last_upper:
            Buy_Signal=False; Sell_Signal=True; signal_reason="매도"
        else:
            Buy_Signal=Sell_Signal=False; signal_reason="관망"

        # ── 단기/장기 전략 ──
        short_strategy = "단기: " + ("매수 추천" if Buy_Signal else "매도 추천" if Sell_Signal else "관망")
        long_strategy = "장기: 상승추세" if last_close>last_ma20 else "장기: 하락추세 또는 관망"

        # ── 기간별 등락률 계산 ──
        periods_for_change = {
            "1일": "2d",
            "1주": "7d",
            "1달": "1mo",
            "3달": "3mo",
            "6달": "6mo",
            "1년": "1y",
            "5년": "5y"
        }
        period_change_pct = {}
        for label, p in periods_for_change.items():
            try:
                hist = yf.download(ticker, period=p, interval="1d", progress=False, auto_adjust=True)
                if not hist.empty:
                    start_close = hist['Close'].iloc[0]
                    end_close = hist['Close'].iloc[-1]
                    pct_change = round((end_close - start_close)/start_close*100,2)
                    period_change_pct[label] = pct_change
            except:
                period_change_pct[label] = None

        # ── 아코디언 UI ──
        with st.expander(f"{ticker} 정보 보기", expanded=True):
            st.markdown(f"**종가:** {last_close:.2f} USD ({last_date})")
            st.markdown(f"**AI Score:** {score} [{grade}]")
            st.markdown(f"**매수/매도 근거:** {reason}")
            st.markdown(f"**평단가 대비 수익률:** {profit_text}")
            st.markdown(f"**Signal:** {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'} ({signal_reason})")
            st.markdown(f"**단기 전략:** {short_strategy}")
            st.markdown(f"**장기 전략:** {long_strategy}")

            st.markdown("### 📊 기간별 등락률")
            for k,v in period_change_pct.items():
                if v is not None:
                    st.markdown(f"- {k}: {v}%")
                else:
                    st.markdown(f"- {k}: 데이터 없음")

            # ── Plotly 차트 ──
            chart_len = 250
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
