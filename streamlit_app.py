import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from polygon import RESTClient

# API 키는 Secrets에서 가져오거나 입력받음 (보안 완벽)
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
    with st.spinner("실시간 데이터 불러오는 중..."):
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
            positive_flow = money_flow.where(typical_price > typical_price.shift(1),
