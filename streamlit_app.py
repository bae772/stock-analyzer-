# app_safe_dashboard_final_v2.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="안전한 주식 투자 대시보드", layout="wide")
st.title("📊 주식 투자 점수 대시보드 (Prophet 없이)")

ticker = st.text_input("주식 티커 입력 (예: AAPL, TSLA, VOO)")
days = st.number_input("최근 추세 기반 예측 기간 (일)", min_value=1, max_value=30, value=5)

if ticker:
    try:
        # 데이터 다운로드
        data = yf.download(ticker, period="5y")
        if data.empty:
            st.warning("데이터를 가져올 수 없습니다. 티커를 확인해주세요.")
        else:
            data = data.copy()
            data['Close'].fillna(method='bfill', inplace=True)

            # 이동평균
            ma20 = data['Close'].rolling(20).mean().fillna(method='bfill')
            ma50 = data['Close'].rolling(50).mean().fillna(method='bfill')
            data['MA20'] = ma20
            data['MA50'] = ma50

            # 볼린저 밴드 (안전하게)
            rolling_std = data['Close'].rolling(20).std().fillna(method='bfill')
            data['BB_upper'] = ma20 + 2 * rolling_std
            data['BB_lower'] = ma20 - 2 * rolling_std

            # RSI
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean().fillna(method='bfill')
            avg_loss = loss.rolling(14).mean().fillna(method='bfill')
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = exp12 - exp26
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # 매수/매도 신호 + 기술 점수
            signal = []
            tech_score = []
            for i in range(len(data)):
                score = 0
                if data['RSI'].iloc[i] < 30 and data['Close'].iloc[i] < data['BB_lower'].iloc[i]:
                    signal.append("매수")
                    score += 1
                elif data['RSI'].iloc[i] > 70 and data['Close'].iloc[i] > data['BB_upper'].iloc[i]:
                    signal.append("매도")
                    score -= 1
                elif data['MACD'].iloc[i] > data['MACD_signal'].iloc[i]:
                    signal.append("매수")
                    score += 0.5
                elif data['MACD'].iloc[i] < data['MACD_signal'].iloc[i]:
                    signal.append("매도")
                    score -= 0.5
                else:
                    signal.append("관망")
                tech_score.append(score)
            data['Signal'] = signal
            data['TechScore'] = tech_score

            # 최근 추세 기반 점수
            if len(data) > days:
                recent_trend = (data['Close'].iloc[-1] - data['Close'].iloc[-days]) / data['Close'].iloc[-days]
                trend_score = 1 if recent_trend > 0 else -1
            else:
                trend_score = 0

            # 뉴스 감성 점수 (예시)
            news_score = 0.5

            # 종합 점수 계산
            data['TotalScore'] = data['TechScore'] + trend_score + news_score
            # 0~10 스케일링
            min_score = data['TotalScore'].min()
            max_score = data['TotalScore'].max()
            if max_score - min_score != 0:
                data['TotalScore'] = ((data['TotalScore'] - min_score) / (max_score - min_score)) * 10
            else:
                data['TotalScore'] = 5
            data['TotalScore'] = data['TotalScore'].fillna(5)

            # 차트
            st.subheader(f"{ticker} 종가 + 매수/매도 신호")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_upper'], mode='lines', name='BB_upper', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_lower'], mode='lines', name='BB_lower', line=dict(dash='dot')))

            buy_signals = data[data['Signal'] == '매수']
            sell_signals = data[data['Signal'] == '매도']
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                     mode='markers', name='매수', marker=dict(color='green', size=10, symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                     mode='markers', name='매도', marker=dict(color='red', size=10, symbol='triangle-down')))
            st.plotly_chart(fig, use_container_width=True)

            # RSI / MACD 차트
            st.subheader("RSI / MACD")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], mode='lines', name='RSI'))
            fig2.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
            fig2.add_trace(go.Scatter(x=data.index, y=data['MACD_signal'], mode='lines', name='MACD_signal'))
            st.plotly_chart(fig2, use_container_width=True)

            # TXT 다운로드
            st.subheader("최근 데이터 다운로드 (TXT)")
            recent_data = data[['Close','MA20','MA50','BB_upper','BB_lower','RSI','MACD','MACD_signal','Signal','TotalScore']].tail(20)
            txt_data = recent_data.to_csv(sep='\t')
            st.download_button(label="TXT 다운로드", data=txt_data, file_name=f"{ticker}_recent.txt", mime="text/plain")

            # 최근 데이터 테이블
            st.subheader("최근 데이터 + 종합 점수")
            st.dataframe(recent_data)

    except Exception as e:
        st.error(f"오류 발생: {e}")
