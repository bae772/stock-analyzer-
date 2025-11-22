# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="키움식 AI 스코어 카드", layout="centered")
st.title("키움증권 스타일 AI 분석 카드 (미국주식)")

ticker = st.text_input("티커 입력 (예: BMR, SLMT, MARA, TSLA)", value="BMR").upper()

if ticker:
    try:
        # 1년치 일봉 데이터
        data = yf.download(ticker, period="1y", progress=False)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커가 잘못되었습니다.")
            st.stop()

        close = data['Close']
        high  = data['High']
