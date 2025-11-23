# -*- coding: utf-8 -*-
"""
한국형 미친 AI 스코어 분석기 v6.66 (지옥에서 온 버전)
라인 수: 현재 1,050줄 넘음 (주석 포함)
작성자: 니들이 알던 그 미친놈
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pyupbit
import ccxt
import talib
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class KoreanMadStockAnalyzer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.binance = ccxt.binance()
        
    def get_data(self, ticker, interval="1d", days=365):
        if ticker.startswith("KRW-") or ticker.startswith("BTC-"):
            df = pyupbit.get_ohlcv(ticker, count=days)
        elif len(ticker) <= 5 and ticker.isalpha():
            df = yf.download(ticker, period=f"{days}d", interval=interval)
        else:
            df = self.get_binance_data(ticker)
        return df.dropna()

    def get_binance_data(self, symbol):
        ohlcv = self.binance.fetch_ohlcv(symbol.replace("/", ""), '1d', limit=500)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def mad_feature_engineering(self, df):
        o, h, l, c, v = df['open'], df['high'], df['low'], df['close'], df['volume']
        
        # 1. 기본 기술적 지표 (TA-Lib 풀 장착)
        df['RSI'] = talib.RSI(c, 14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(c)
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(c)
        df['ATR'] = talib.ATR(h, l, c, 14)
        df['ADX'] = talib.ADX(h, l, c, 14)
        df['CCI'] = talib.CCI(h, l, c, 20)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(h, l, c)
        df['WILLR'] = talib.WILLR(h, l, c, 14)
        df['MFI'] = talib.MFI(h, l, c, v, 14)
        df['OBV'] = talib.OBV(c, v)

        # 2. 미친놈들의 단기 폭발력 지표
        df['gap'] = (o / c.shift(1) - 1) * 100
        df['body'] = abs(c - o) / o * 100
        df['upper_shadow'] = (h - np.maximum(c, o)) / o * 100
        df['lower_shadow'] = (np.minimum(c, o) - l) / o * 100
        df['range'] = (h - l) / o * 100
        
        # 3. 거래량 폭발
        df['vol_ratio'] = v / v.rolling(20).mean()
        df['vol_shock'] = (v / v.shift(1)).apply(lambda x: 10 if x > 10 else x)
        
        # 4. 단기 급등 패턴 (한국형)
        df['pump_3d'] = c / c.shift(3) - 1
        df['pump_5d'] = c / c.shift(5) - 1
        df['golden_cross'] = (df['close'].rolling(5).mean() > df['close'].rolling(20).mean()) & \
                              (df['close'].rolling(5).mean().shift(1) <= df['close'].rolling(20).mean().shift(1))
        
        # 5. 외인/기관 추정 (미국 주식은 의미 없지만 한국형이라 넣음)
        df['foreign_buy'] = np.random.uniform(-1, 3, len(df))  # 실제론 증권사 API 필요

        # 6. VIX 연동 공포지수 반영 (미국 주식 필수)
        try:
            vix = yf.Ticker("^VIX").history(period="5d")['Close'].iloc[-1]
            df['vix_fear'] = 100 if vix > 40 else 80 if vix > 30 else 50 if vix > 20 else 10
        except:
            df['vix_fear'] = 30

        return df

    def calculate_ai_score(self, ticker):
        df = self.get_data(ticker)
        if len(df) < 50:
            return {"ticker": ticker, "AI_SCORE": 0, "signal": "데이터 부족"}
            
        df = self.mad_feature_engineering(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        score = 50  # 기본점수

        # 1. 단기 급등 (30점 만점)
        if latest['pump_3d'] > 0.15: score += 25
        elif latest['pump_3d'] > 0.08: score += 18
        elif latest['pump_3d'] > 0.05: score += 10

        # 2. 거래량 폭발 (20점)
        if latest['vol_ratio'] > 5: score += 20
        elif latest['vol_ratio'] > 3: score += 15
        elif latest['vol_ratio'] > 2: score += 8

        # 3. 골든크로스 + 상승장 (15점)
        if latest['golden_cross']: score += 15
        if latest['RSI'] < 70 and prev['RSI'] >= 70: score += 10  # 과매도 탈출

        # 4. 캔들 패턴 (15점)
        if latest['body'] > 5 and latest['lower_shadow'] < 1: score += 15  # 장대양봉
        if latest['gap'] > 5: score += 12  # 갭상승

        # 5. 모멘텀 + 재료 (20점)
        if latest['MFI'] > 80 and latest['RSI'] < 70: score += 20  # 돈 들어오는데 안 오른 놈
        if latest['OBV'] > latest['OBV'].rolling(20).max(): score += 15

        # 6. 공포탐욕 차단 (감점)
        if latest['vix_fear'] > 80: score -= 20

        score = max(0, min(100, score))

        # 최종 등급
        if score >= 90: grade = "SSSS"
        elif score >= 85: grade = "SSS"
        elif score >= 80: grade = "SS"
        elif score >= 70: grade = "S"
        elif score >= 60: grade = "A"
        else: grade = "B 이하"

        target_price = latest['close'] * (1 + score/1000)
        stop_loss = latest['close'] * 0.93

        return {
            "ticker": ticker,
            "price": round(latest['close'], 4),
            "AI_SCORE": int(score),
            "등급": grade,
            "추천": "상승장(강함)" if score >= 70 else "관망" if score >= 50 else "위험",
            "
