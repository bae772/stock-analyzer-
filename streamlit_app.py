# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, CCIIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Ultimate Stock Analyzer", layout="wide")
st.title("📈 Ultimate Stock Analyzer - All-in-One")

# ---------------------------
# 사용자 입력
# ---------------------------
ticker = st.text_input("종목 코드 (예: AAPL)", "AAPL")
start_date = st.date_input("시작일", datetime.now() - timedelta(days=730))
end_date = st.date_input("종료일", datetime.now())

# ---------------------------
# 데이터 가져오기
# ---------------------------
@st.cache_data
def get_stock_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

df = get_stock_data(ticker, start_date, end_date)

# ---------------------------
# 기술적 지표 계산
# ---------------------------
def compute_indicators(df):
    # 이동평균 및 EMA
    for w in [5,10,20,50,100,200]:
        df[f'SMA{w}'] = SMAIndicator(df['Close'], window=w).sma_indicator()
        df[f'EMA{w}'] = EMAIndicator(df['Close'], window=w).ema_indicator()
    # 모멘텀
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    df['Stochastic'] = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch()
    df['ROC'] = ROCIndicator(df['Close'], window=12).roc()
    # 트렌드
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['ADX'] = ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
    df['CCI'] = CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci()
    ichimoku = IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_base'] = ichimoku.ichimoku_base_line()
    # 변동성
    bb = BollingerBands(df['Close'], window=20)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_low'] = bb.bollinger_lband()
    df['ATR'] = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['KC_high'] = kc.keltner_channel_hband()
    df['KC_low'] = kc.keltner_channel_lband()
    # 거래량
    df['OBV'] = OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['CMF'] = ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow()
    df['VWAP'] = VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14).volume_weighted_average_price()
    return df

df = compute_indicators(df)

# ---------------------------
# 캔들 패턴 탐지
# ---------------------------
def detect_candles(df):
    df['Bullish_Engulfing'] = ((df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))).astype(int)
    df['Bearish_Engulfing'] = ((df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))).astype(int)
    df['Hammer'] = ((df['Close'] > df['Open']) & ((df['Low'] - df[['Open','Close']].min(axis=1)) / (df['High'] - df['Low']) > 0.6)).astype(int)
    df['Shooting_Star'] = ((df['Close'] < df['Open']) & ((df['High'] - df[['Open','Close']].max(axis=1)) / (df['High'] - df['Low']) > 0.6)).astype(int)
    return df

df = detect_candles(df)

# ---------------------------
# AI 점수 계산
# ---------------------------
df.dropna(inplace=True)
features = ['SMA20','SMA50','EMA20','RSI','Stochastic','ROC','MACD','MACD_signal','ADX','CCI','Ichimoku_base',
            'BB_high','BB_low','ATR','KC_high','KC_low','OBV','CMF','VWAP',
            'Bullish_Engulfing','Bearish_Engulfing','Hammer','Shooting_Star']

X = df[features]
y = (df['Close'].shift(-1) > df['Close']).astype(int)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

df['AI_Score'] = model.predict_proba(X_scaled)[:,1]*100
df['Signal'] = np.where(df['AI_Score']>60,'Buy',np.where(df['AI_Score']<40,'Sell','Hold'))
df['Target'] = df['Close']*(1 + df['AI_Score']/100*0.05)
df['Stop_Loss'] = df['Close']*(1 - df['AI_Score']/100*0.03)

# ---------------------------
# UI
