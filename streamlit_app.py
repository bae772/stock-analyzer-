# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ta 라이브러리 임포트
try:
    from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
    from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, CCIIndicator, IchimokuIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator, VolumeWeightedAveragePrice
except ModuleNotFoundError as e:
    st.error(f"필요한 라이브러리가 설치되지 않았습니다: {e.name}. requirements.txt 확인")
    st.stop()

st.set_page_config(page_title="Ultimate Stock Analyzer", layout="wide")
st.title("📈 Ultimate Stock Analyzer - Fully Fixed Version")

# ---------------------------
# 사용자 입력
ticker = st.text_input("종목 코드", "AAPL")
start_date = st.date_input("시작일", datetime.now() - timedelta(days=730))
end_date = st.date_input("종료일", datetime.now())

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)
    return df

df = load_data(ticker, start_date, end_date)

# ---------------------------
# 기술적 지표
def compute_indicators(df):
    for w in [5,10,20,50,100,200]:
        df[f"SMA{w}"] = pd.Series(SMAIndicator(df['Close'], window=w).sma_indicator().to_numpy().flatten(), index=df.index)
        df[f"EMA{w}"] = pd.Series(EMAIndicator(df['Close'], window=w).ema_indicator().to_numpy().flatten(), index=df.index)
    df['RSI'] = pd.Series(RSIIndicator(df['Close'], window=14).rsi().to_numpy().flatten(), index=df.index)
    df['Stochastic'] = pd.Series(StochasticOscillator(df['High'], df['Low'], df['Close'], window=14).stoch().to_numpy().flatten(), index=df.index)
    df['ROC'] = pd.Series(ROCIndicator(df['Close'], window=12).roc().to_numpy().flatten(), index=df.index)
    
    macd = MACD(df['Close'])
    df['MACD'] = pd.Series(macd.macd().to_numpy().flatten(), index=df.index)
    df['MACD_signal'] = pd.Series(macd.macd_signal().to_numpy().flatten(), index=df.index)
    
    df['ADX'] = pd.Series(ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx().to_numpy().flatten(), index=df.index)
    df['CCI'] = pd.Series(CCIIndicator(df['High'], df['Low'], df['Close'], window=20).cci().to_numpy().flatten(), index=df.index)
    
    ichimoku = IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_base'] = pd.Series(ichimoku.ichimoku_base_line().to_numpy().flatten(), index=df.index)
    
    bb = BollingerBands(df['Close'], window=20)
    df['BB_high'] = pd.Series(bb.bollinger_hband().to_numpy().flatten(), index=df.index)
    df['BB_low'] = pd.Series(bb.bollinger_lband().to_numpy().flatten(), index=df.index)
    
    df['ATR'] = pd.Series(AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range().to_numpy().flatten(), index=df.index)
    
    kc = KeltnerChannel(df['High'], df['Low'], df['Close'], window=20)
    df['KC_high'] = pd.Series(kc.keltner_channel_hband().to_numpy().flatten(), index=df.index)
    df['KC_low'] = pd.Series(kc.keltner_channel_lband().to_numpy().flatten(), index=df.index)
    
    df['OBV'] = pd.Series(OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume().to_numpy().flatten(), index=df.index)
    df['CMF'] = pd.Series(ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=20).chaikin_money_flow().to_numpy().flatten(), index=df.index)
    df['VWAP'] = pd.Series(VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume'], window=14).volume_weighted_average_price().to_numpy().flatten(), index=df.index)
    
    return df

df = compute_indicators(df)

# ---------------------------
# 캔들패턴 일부 예제 (50종 확장 가능)
df['Bullish_Engulfing'] = ((df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1))).astype(int)
df['Bearish_Engulfing'] = ((df['Close'] < df['Open'].shift(1)) & (df['Open'] > df['Close'].shift(1))).astype(int)
df['Hammer'] = ((df['Close'] > df['Open']) & ((df['Low'] - df[['Open','Close']].min(axis=1)) / (df['High'] - df['Low']) > 0.6)).astype(int)
df['Shooting_Star'] = ((df['Close'] < df['Open']) & ((df['High'] - df[['Open','Close']].max(axis=1)) / (df['High'] - df['Low']) > 0.6)).astype(int)

# ---------------------------
# AI 점수 계산
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
# UI 탭 및 차트
tab1, tab2, tab3, tab4 = st.tabs(["차트","지표","데이터","백테스트"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], line=dict(color='blue', width=1), name='SMA20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], line=dict(color='orange', width=1), name='SMA50'))
    buy_signals = df[df['Signal']=='Buy']
    sell_signals = df[df['Signal']=='Sell']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker_symbol='triangle-up', marker_color='green', name='Buy'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker_symbol='triangle-down', marker_color='red', name='Sell'))
    fig.update_layout(title=f"{ticker} 차트 및 신호", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["RSI","MACD","ADX"])
    fig2.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='cyan'), name='RSI'), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='orange'), name='MACD'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='blue'), name='MACD_signal'), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df.index, y=df['ADX'], line=dict(color='yellow'), name='ADX'), row=3, col=1)
    fig2.update_layout(template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.dataframe(df.tail(50))

with tab4:
    st.write("백테스트 기능 예시 (확장 가능)")
    st.line_chart(df['Close'].pct_change().cumsum())

st.subheader("최근 전략 요약")
st.write(f"AI 점수: {df['AI_Score'].iloc[-1]:.2f}, 신호: {df['Signal'].iloc[-1]}, 목표가: {df['Target'].iloc[-1]:.2f}, 손절가: {df['Stop_Loss'].iloc[-1]:.2f}")
