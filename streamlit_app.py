# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="간단 주식 신호 앱", layout="wide")
st.title("📈 주식 신호 & 점수 대시보드 (초간단 버전)")

ticker = st.text_input("주식 티커 입력 (예: AAPL, TSLA, VOO)")

if ticker:
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            st.warning("데이터를 가져올 수 없습니다.")
        else:
            data['Close'].fillna(method='bfill', inplace=True)

            # 단순 이동평균
            ma20 = data['Close'].rolling(20).mean().iloc[-1]
            ma50 = data['Close'].rolling(50).mean().iloc[-1]
            last_close = data['Close'].iloc[-1]

            # RSI 계산 (최근 값만)
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))

            # 매수/매도 신호
            if rsi < 30:
                signal = "매수"
            elif rsi > 70:
                signal = "매도"
            else:
                signal = "관망"

            # 점수 계산 (간단)
            trend_score = (last_close / data['Close'].iloc[-6]) * 50
            rsi_score = max(0, 100 - abs(50 - rsi))
            total_score = min(100, trend_score + rsi_score)
            if total_score >= 95:
                grade = "SSS"
            elif total_score >= 90:
                grade = "SS"
            elif total_score >= 80:
                grade = "S"
            elif total_score >= 70:
                grade = "A"
            elif total_score >= 60:
                grade = "B"
            else:
                grade = "C"

            # 추세, 패턴, 목표/손절
            trend = "상승" if last_close > data['Close'].iloc[-6] else "하락"
            pattern = "상향" if ma20 > ma50 else "하향"
            target = round(last_close * 1.05, 2)
            stop_loss = round(last_close * 0.95, 2)
            caution = "최근 변동성 주의"

            # 요약 카드
            st.subheader(f"{ticker} 요약")
            st.markdown(f"- **현재가:** {last_close:.2f}")
            st.markdown(f"- **신호:** {signal}")
            st.markdown(f"- **점수:** {total_score:.1f}/100 ({grade})")
            st.markdown(f"- **추세:** {trend}")
            st.markdown(f"- **패턴:** {pattern}")
            st.markdown(f"- **목표:** {target}")
            st.markdown(f"- **손절:** {stop_loss}")
            st.markdown(f"- **주의사항:** {caution}")

            # 차트 (간단)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(20).mean(), mode='lines', name='MA20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'].rolling(50).mean(), mode='lines', name='MA50'))
            st.plotly_chart(fig, use_container_width=True)

            # 최근 데이터 + TXT 다운로드
            recent_data = data[['Close']].tail(20)
            st.subheader("최근 데이터")
            st.dataframe(recent_data)
            txt_data = recent_data.to_csv(sep='\t')
            st.download_button(label="TXT 다운로드", data=txt_data, file_name=f"{ticker}_recent.txt", mime="text/plain")

    except Exception as e:
        st.error(f"오류 발생: {e}")
