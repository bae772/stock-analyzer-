# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import os

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기")

# ── 자동 학습 함수 ──
history_file = "prediction_history.csv"
def auto_learn(history_csv=history_file):
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

score_weights = auto_learn(history_file)

# ── 사용자 입력 ──
tickers = st.text_input("티커 입력 (쉼표로 여러개 입력 가능, 예: BMR, TSLA, MARA)", value="BMR").upper().replace(" ", "").split(",")
avg_price = st.number_input("보유 평단가 입력 (없으면 0)", min_value=0.0, step=0.01, value=0.0)

# ── 기간 선택 ──
period_options = {
    "실시간": "1d",
    "1일": "1d",
    "1주": "7d",
    "3달": "3mo",
    "1년": "1y",
    "5년": "5y",
    "10년": "10y",
    "15년": "15y"
}
selected_period = st.selectbox("기간 선택", list(period_options.keys()))

for ticker in tickers:
    if not ticker: continue
    try:
        data = yf.download(ticker, period=period_options[selected_period], progress=False, auto_adjust=True)
        if data.empty or len(data) < 2:
            st.warning(f"{ticker}: 데이터가 부족하거나 티커를 찾을 수 없습니다.")
            continue
        data = data[['Close','High','Low','Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close-prev_close)/prev_close*100,2)
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
        vol_ratio = round(vol_today / last_vol_ma20, 2) if last_vol_ma20>0 else 1.0

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
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5,2)
        stop_price = round(last_close - last_atr*1.8,2)
        target_pct = round((target_price - last_close)/last_close*100,1)
        stop_pct = round((stop_price - last_close)/last_close*100,1)

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

        # 등급
        if score>=80: grade, gcolor, reason = "A (강력매수)","#00ff00","RSI 저평가 + 상승추세 + MACD 골든크로스"
        elif score>=70: grade, gcolor, reason = "A (매수)","#33ff33","추세 상승 + 일부 기술적 지표 긍정적"
        elif score>=60: grade, gcolor, reason = "B (관망)","#ffff33","단기 변동성 존재, 신중 관망 필요"
        elif score>=40: grade, gcolor, reason = "C (주의)","#ff9933","과열/하락 위험, 일부 매수 가능성만"
        else: grade, gcolor, reason = "D (매도)","#ff3333","과열/하락 신호 다수, 매수 지양"

        # 평단 대비
        if avg_price>0:
            profit_pct = round((last_close-avg_price)/avg_price*100,2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct>=0 else '손실'})"
            if last_close>=target_price: sell_advice="목표가 도달! 매도 고려"
            elif last_rsi>70: sell_advice="RSI 과열, 단기 매도 가능"
            elif last_close<stop_price: sell_advice="손절가 도달, 손절 권장"
            elif last_ma20>last_ma50: sell_advice="장기 상승추세 유지, 보유 추천"
            else: sell_advice="단기 변동성 높음, 추세 확인 후 판단"
        else:
            profit_text="평단가 입력 없음"
            sell_advice="평단가 미입력, 매도 전략 판단 불가"

        # Signal
        if last_rsi<30 and last_macd_diff>0 and last_close<last_lower:
            Buy_Signal=True; Sell_Signal=False; signal_reason="강력 매수: RSI 매우 저평가 + MACD 골든크로스 + 볼린저 하단"
        elif last_rsi<40 and last_macd_diff>0:
            Buy_Signal=True; Sell_Signal=False; signal_reason="매수: RSI 저평가 + MACD 골든크로스"
        elif last_rsi>70 or last_close>last_upper:
            Buy_Signal=False; Sell_Signal=True; signal_reason="매도: RSI 과열 + 볼린저 상단 돌파"
        else:
            Buy_Signal=False; Sell_Signal=False; signal_reason="관망: 단기 신호 불확실"

        # 단기/장기 전략
        if Buy_Signal and vol_ratio>1.5: short_strategy="단기: 변동성 급등 구간 소량 매수 추천"
        elif Buy_Signal: short_strategy="단기: 일반 매수 구간, 소량 매수 가능"
        elif Sell_Signal and vol_ratio>1.5: short_strategy="단기: 과열 구간, 일부 매도 권장"
        elif Sell_Signal: short_strategy="단기: 매도 신호, 비중 일부 조정"
        else: short_strategy="단기: 관망, 신호 불확실"

        if last_close>last_ma20 and last_ma20>last_ma50: long_strategy="장기: 상승추세, 비중 확대 가능"
        elif last_close<last_ma20 and last_ma20<last_ma50: long_strategy="장기: 하락추세, 신규 매수 지양"
        else: long_strategy="장기: 관망, 추세 확인 필요"

        # ── 카드 출력 ──
        st.markdown(f"""
        <div style="background:#000; color:white; padding:20px; border-radius:15px; 
                    text-align:center; border:2px solid #00ffcc; box-shadow:0 0 20px #00ffcc99;">
            <h1 style="color:#00ffcc;">{ticker}</h1>
            <h2>${last_close:.2f} ({last_date})</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'};">{'+' if change_pct>=0 else ''}{change_pct}%</p>
            <h3 style="color:{gcolor};">등급: {grade}</h3>
            <p>AI 근거: {reason}</p>
            <p>평단 대비: {profit_text}</p>
            <p>판매 안내: {sell_advice}</p>
            <p>단기 전략: {short_strategy}</p>
            <p>장기 전략: {long_strategy}</p>
            <p>Signal: {'BUY' if Buy_Signal else 'SELL' if Sell_Signal else 'HOLD'}</p>
            <p>Signal 근거: {signal_reason}</p>
        </div>
        """, unsafe_allow_html=True)

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

        # ── 날짜별 수익률 ──
        returns = close.pct_change().fillna(0)*100
        returns_df = pd.DataFrame({
            "날짜": close.index.strftime('%Y-%m-%d'),
            "종가": close.values,
            "수익률(%)": returns.round(2)
        })
        st.markdown("**최근 날짜별 수익률 (전일 대비)**")
        st.dataframe(returns_df.tail(30))

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        st.error(f"{ticker} 오류 발생: {e}")
