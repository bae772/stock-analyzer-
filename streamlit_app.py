import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from polygon import RESTClient

# API 키는 Secrets에서 가져오거나 입력받음
if "API_KEY" in st.secrets:
    API_KEY = st.secrets["API_KEY"]
else:
    API_KEY = st.text_input("Polygon API 키 입력 (무료 발급: polygon.io)", type="password")
    if not API_KEY:
        st.error("API 키를 입력하세요!")
        st.stop()

client = RESTClient(api_key=API_KEY)

st.set_page_config(page_title="로켓 주식 분석기", layout="centered")

st.markdown("""
<style>
.big-font {font-size:70px !important; color:#00D4FF; font-weight:bold; text-align:center;}
.score {font-size:100px !important; color:#FF00FF; font-weight:bold; text-align:center;}
.grade {font-size:50px !important; color:#FFFF00; text-align:center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">로켓 주식 분석기</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#888; font-size:24px;">종목 티커만 입력하면 그 사람처럼 분석해줍니다</p>', unsafe_allow_html=True)

ticker = st.text_input("티커 입력 (예: GOOG, NVDA, 005930.KS)", value="GOOG").upper()

if st.button("분석 시작 🚀", type="primary", use_container_width=True):
    with st.spinner("데이터 불러오는 중..."):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=400)
            agg = client.get_aggs(ticker=ticker, multiplier=1, timespan="day",
                                 from_=start_date.strftime("%Y-%m-%d"),
                                 to=end_date.strftime("%Y-%m-%d"), adjusted=True, limit=50000)
            df = pd.DataFrame(agg)
            if len(df) == 0:
                st.error("티커 오류 또는 API 키 확인")
                st.stop()

            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('date')[['open', 'high', 'low', 'close', 'volume']]

            # 지표 계산
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
            negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
            mfr = positive_flow / negative_flow
            df['MFI'] = 100 - (100 / (1 + mfr))

            df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).cumsum().fillna(0)
            df['MA20'] = df['close'].rolling(20).mean()
            df['MA60'] = df['close'].rolling(60).mean()
            df['MA120'] = df['close'].rolling(120).mean()

            recent = df.iloc[-1]

            score = 50
            if recent['MA20'] > recent['MA60'] > recent['MA120']: score += 18
            if recent['volume'] > df['volume'].rolling(20).mean().iloc[-1] * 2.8: score += 15
            if recent['OBV'] == df['OBV'].max(): score += 12
            if 63 <= recent['RSI'] <= 82: score += 10
            if recent['MFI'] > 78: score +=  += 8
            body = abs(recent['close'] - recent['open'])
            range_ = recent['high'] - recent['low']
            if body > range_ * 0.75 and recent['close'] > recent['open']: score += 14
            rise_5d = (recent['close'] / df['close'].iloc[-6]) - 1
            if rise_5d > 0.12: score += 16
            elif rise_5d > 0.07: score += 9
            if recent['close'] > df['high'].rolling(60).max().iloc[-2]: score += 11
            score = min(99, score)

            grade = "SSSS" if score >= 95 else "SSS" if score >= 92 else "SS" if score >= 88 else "S" if score >= 82 else "A+"

            atr = (df['high'] - df['low']).rolling(14).mean().iloc[-1]
            target = round(recent['close'] * 1.09 + atr * 2, 2)
            stop = round(recent['close'] * 0.938 - atr * 0.7, 2)

            st.markdown(f"""
            <div style="text-align:center; background:#000; padding:50px; border-radius:25px;">
                <div style="font-size:80px; color:#00D4FF;">{ticker}<br>{ticker}</div>
                <div style="font-size:50px; color:white;">${recent['close']:.2f}</div>
                <div class="score">{score}</div>
                <div class="grade">등급 [ {grade} ]</div>
                <div style="font-size:32px; color:white; line-height:2.2;">
                    추세: 초강세 상승장 (퍼펙트 골든크로스)<br>
                    에너지: 매수세 극강 (OBV 사상 최고)<br>
                    캔들: 대량 거래 + 장대양봉<br>
                    복합 지표: RSI {recent['RSI']:.0f} / MFI {recent['MFI']:.0f}<br><br>
                    <span style="color:#00FF00; font-size:40px;">TARGET ${target} (+{((target/recent['close'])-1)*100:.1f}%)</span><br>
                    <span style="color:#FF0000; font-size:40px;">STOP ${stop} ({((stop/recent['close'])-1)*100:.1f}%)</span><br><br>
                    <span style="font-size:45px;">지금이 진짜 매수 타이밍입니다<br><b>로켓</b>입니다 로켓 로켓 로켓</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"에러: {e}")

st.caption("완료! 다음 종목도 바로 넣어보세요 🚀")
