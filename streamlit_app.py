# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이 계산기", layout="centered")
st.title("베짱이 계산기 (미국주식 AI 분석 + 투자 전략)")

# ── 단일 종목 분석 ──
ticker = st.text_input("티커 입력 (예: BMR, TSLA, MARA)", value="BMR").upper()
avg_price = st.number_input("보유 평단가 입력 (없으면 0)", min_value=0.0, step=0.01, value=0.0)

# 날짜 선택
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("시작일", value=pd.to_datetime("2025-01-01"))
with col2:
    end_date = st.date_input("종료일", value=pd.to_datetime(datetime.today().date()))

# ── 단일 종목 계산 함수 ──
def analyze_stock(ticker, avg_price, start_date=None, end_date=None):
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1),
                               progress=False, auto_adjust=True)
        else:
            data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)

        if data.empty or len(data) < 50:
            return None
        data = data[['Close','High','Low','Volume']].dropna()
        close, high, low, volume = data['Close'], data['High'], data['Low'], data['Volume']

        # 기본 지표 계산
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close)/prev_close*100, 2)
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
        tr = pd.concat([tr1,tr2,tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1])
        target_price = round(last_close + last_atr*2.5,2)
        stop_price = round(last_close - last_atr*1.8,2)

        # AI SCORE
        score = 50.0
        score += max(0,30-last_rsi)*1.4
        score += change_pct*2
        score += max(0,vol_ratio-1)*12
        score += 15 if last_close>last_ma20 else -10
        score += 10 if last_close>last_ma50 else -8
        score += 10 if last_macd_diff>0 else -10
        score += 5 if last_close<last_lower else -5
        score = int(np.clip(score,0,100))

        # 등급 + 근거
        if score>=80:
            grade, reason = "A (강력매수)", "RSI 저평가 + 상승추세 + MACD 골든크로스 등 긍정적 신호"
        elif score>=70:
            grade, reason = "A (매수)", "추세 상승 + 일부 기술적 지표 긍정적"
        elif score>=60:
            grade, reason = "B (관망)", "단기 변동성 존재, 신중 관망 필요"
        elif score>=40:
            grade, reason = "C (주의)", "과열/하락 위험, 일부 매수 가능성만"
        else:
            grade, reason = "D (매도)", "과열/하락 신호 다수, 매수 지양"

        # 평단가 대비 수익률
        profit_pct = round((last_close-avg_price)/avg_price*100,2) if avg_price>0 else None

        # Signal
        if last_rsi < 30 and last_macd_diff > 0 and last_close < last_lower:
            Buy_Signal = True; Sell_Signal = False; signal_reason = "강력 매수: RSI 매우 저평가 + MACD 골든크로스 + 볼린저밴드 하단"
        elif last_rsi < 40 and last_macd_diff > 0:
            Buy_Signal = True; Sell_Signal = False; signal_reason = "매수: RSI 저평가 + MACD 골든크로스"
        elif last_rsi > 70 or last_close > last_upper:
            Buy_Signal = False; Sell_Signal = True; signal_reason = "매도: RSI 과열 + 볼린저밴드 상단 돌파"
        else:
            Buy_Signal = Sell_Signal = False; signal_reason = "관망: 단기 신호 불확실"

        # 단기/장기 전략
        if Buy_Signal and vol_ratio > 1.5:
            short_strategy = "단기: 변동성 급등 구간 소량 매수 추천"
        elif Buy_Signal:
            short_strategy = "단기: 일반 매수 구간, 소량 매수 가능"
        elif Sell_Signal and vol_ratio > 1.5:
            short_strategy = "단기: 과열 구간, 일부 매도 권장"
        elif Sell_Signal:
            short_strategy = "단기: 매도 신호, 비중 일부 조정"
        else:
            short_strategy = "단기: 관망, 신호 불확실"

        if last_close > last_ma20 and last_ma20 > last_ma50:
            long_strategy = "장기: 상승추세, 비중 확대 가능"
        elif last_close < last_ma20 and last_ma20 < last_ma50:
            long_strategy = "장기: 하락추세, 신규 매수 지양"
        else:
            long_strategy = "장기: 관망, 추세 확인 필요"

        # 매도 안내
        if avg_price>0:
            if last_close >= target_price:
                sell_advice = "목표가 도달! 매도 고려"
            elif last_rsi > 70:
                sell_advice = "RSI 과열, 단기 매도 가능"
            elif last_close < stop_price:
                sell_advice = "손절가 도달, 손절 권장"
            elif last_ma20 > last_ma50:
                sell_advice = "장기 상승추세 유지, 보유 추천"
            else:
                sell_advice = "단기 변동성 높음, 추세 확인 후 판단"
        else:
            sell_advice = "평단가 미입력, 매도 전략 판단 불가"

        return {
            "ticker":ticker,
            "last_close":last_close,
            "last_date":last_date,
            "score":score,
            "grade":grade,
            "reason":reason,
            "profit_pct":profit_pct,
            "Buy_Signal":Buy_Signal,
            "Sell_Signal":Sell_Signal,
            "signal_reason":signal_reason,
            "short_strategy":short_strategy,
            "long_strategy":long_strategy,
            "target_price":target_price,
            "stop_price":stop_price
        }

    except Exception as e:
        st.error(f"{ticker} 분석 중 오류: {e}")
        return None

# ── 단일 종목 출력 ──
if ticker:
    result = analyze_stock(ticker, avg_price, start_date, end_date)
    if result:
        st.markdown(f"""
        <div style="background:#000; color:white; padding:30px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:4.5em;">{result['ticker']}</h1>
            <h2 style="margin:10px 0; font-size:3em;">${result['last_close']:.2f} ({result['last_date']})</h2>
            <h3 style="color:{'#00ff00' if result['score']>=75 else '#ffff00' if result['score']>=60 else '#ff9933'};">
                AI SCORE: {result['score']} [{result['grade']}]</h3>
            <p>매수/매도 근거: {result['reason']}</p>
            <p>평단가 대비 수익률: {result['profit_pct'] if result['profit_pct'] is not None else '평단가 입력 없음'}%</p>
            <p>판매 안내: {result['Sell_Signal']} - {result['signal_reason']}</p>
            <p>단기 전략: {result['short_strategy']}</p>
            <p>장기 전략: {result['long_strategy']}</p>
            <p>Target: ${result['target_price']}, Stop Loss: ${result['stop_price']}</p>
        </div>
        """, unsafe_allow_html=True)

# ── 멀티 종목 분석 기능 ──
st.markdown("---")
st.subheader("💡 여러 종목 한 번에 분석")
tickers_multi = st.text_area("분석할 티커 입력 (쉼표로 구분)", value="BMR, TSLA, MARA")
avg_prices_multi = st.text_area("각 티커 평단가 입력 (쉼표 구분, 없으면 0)", value="0,0,0")

tickers_list = [t.strip().upper() for t in tickers_multi.split(",")]
avg_prices_list = [float(p.strip()) if p.strip() else 0 for p in avg_prices_multi.split(",")]
if len(avg_prices_list) < len(tickers_list):
    avg_prices_list += [0]*(len(tickers_list)-len(avg_prices_list))

results_multi = []
for i, t in enumerate(tickers_list):
    res = analyze_stock(t, avg_prices_list[i])
    if res:
        results_multi.append(res)

# 점수 기준 내림차순 정렬
results_multi = sorted(results_multi, key=lambda x:x['score'], reverse=True)

for r in results_multi:
    st.markdown(f"""
    <div style="background:#111; color:white; padding:20px; border-radius:15px; margin-bottom:20px;">
        <h2 style="color:#00ffcc;">{r['ticker']} (${r['last_close']:.2f})</h2>
        <h3 style="color:{'#00ff00' if r['score']>=75 else '#ffff00' if r['score']>=60 else '#ff9933'};">AI SCORE: {r['score']} [{r['grade']}]</h3>
        <p>매수/매도 근거: {r['reason']}</p>
        <p>평단가 대비 수익률: {r['profit_pct'] if r['profit_pct'] is not None else '평단가 입력 없음'}%</p>
        <p>Signal: {'BUY' if r['Buy_Signal'] else 'SELL' if r['Sell_Signal'] else 'HOLD'} - {r['signal_reason']}</p>
        <p>단기 전략: {r['short_strategy']}</p>
        <p>장기 전략: {r['long_strategy']}</p>
        <p>Target: ${r['target_price']}, Stop Loss: ${r['stop_price']}</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")
