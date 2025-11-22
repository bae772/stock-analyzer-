import streamlit as st
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from polygon import RESTClient

# API 키 Secrets 또는 입력받기
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
.big-font {font
