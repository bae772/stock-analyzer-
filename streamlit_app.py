# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="주식 신호 앱", layout="wide")
st.title("📈 주식 신호 & 점수 대시보드 (최종 안정판)")

ticker = st.text_input("주식 티커 입력 (예: AAPL, TSLA, VOO)")
days = st.number_input("최근 추세 기간 (일)", min_value=1, max_value=30, value=5)

def score_to_grade(score):
    if score >= 95: return "SSS"
    if score >= 90: return "SS"
    if score >= 80: return "S"
    if score >= 70: return "A"
    if score >= 60: return "B"
    return "C"

if ticker:
    try:
        data = yf.download(ticker, period="5y")
        if data.empty:
            st.warning("데이터를 가져올 수 없습니다.")
        else:
            data['Close'].fillna(method='bfill', inplace=True)
            data['Volume'].fillna(0, inplace=True)

            # 이동평균
            data['MA20'] = data['Close'].rolling(20).mean().fillna(method='bfill')
            data['MA50'] = data['Close'].rolling(50).mean().fillna(method='bfill')

            # RSI 계산
            delta = data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -1 * delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean().fillna(method='bfill')
            avg_loss = loss.rolling(14).mean().fillna(method='bfill')
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # OBV 계산
            data['OBV'] = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()

            # 매수/매도 신호
            data['Signal'] = "관망"
            data.loc[data['RSI'] < 30, 'Signal'] = "매수"
            data.loc[data['RSI'] > 70, 'Signal'] = "매도"

            # 점수 계산
            trend_score = ((data['Close'].iloc[-1] - data['Close'].iloc[-days]) / data['Close'].iloc[-days]) * 50
            rsi_score = max(0, 100 - abs(50 - data['RSI'].iloc[-1]))
            total_score = min(100, trend_score + rsi_score)
            grade = score_to_grade(total_score)

            # 목표/손절/주의사항
            last_price = data['Close'].iloc[-1]
            target = round(last_price * 1.05, 2)
            stop_loss = round(last_price * 0.95, 2)
            caution = "최근 변동성 높음, 분할 매수 권장"

            # 패턴, 추세
            pattern = "상향" if data['MA20'].iloc[-1] > data['MA20'].iloc[-2] else "하향"
            trend = "상승" if trend_score > 0 else "하락"

            # 요약 카드
            st.subheader(f"{ticker} 요약")
            st.markdown(f"- **현재가:** {last_price:.2f}")
            st.markdown(f"- **신호:** {data['Signal'].iloc[-1]}")
            st.markdown(f"- **점수:** {total_score:.1f}/100 ({grade})")
            st.markdown(f"- **추세:** {trend}")
            st.markdown(f"- **에너지:** RSI={data['RSI'].iloc[-1]:.1f}")
            st.markdown(f"- **패턴:** {pattern}")
            st.markdown(f"- **OBV 잔존률:** {data['OBV'].iloc[-1]:.0f}")
            st.markdown(f"- **목표:** {target}")
            st.markdown(f"- **손절:** {stop_loss}")
            st.markdown(f"- **주의사항:** {caution}")

            # 차트
            st.subheader(f"{ticker} 종가 + 매수/매도 신호")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='MA20', line=dict(color='orange')))
            fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], mode='lines', name='MA50', line=dict(color='green')))
            buy = data[data['Signal']=='매수']
            sell = data[data['Signal']=='매도']
            fig.add_trace(go.Scatter(x=buy.index, y=buy['Close'], mode='markers', name='매수',
                                     marker=dict(color='green', size=10, symbol='triangle-up')))
            fig.add_trace(go.Scatter(x=sell.index, y=sell['Close'], mode='markers', name='매도',
                                     marker=dict(color='red', size=10, symbol='triangle-down')))
            fig.update_layout(height=500, margin=dict(l=20,r=20,t=30,b=20))
            st.plotly_chart(fig, use_container_width=True)

            # 최근 데이터 + 다운로드
            recent_data = data[['Close','MA20','MA50','RSI','OBV','Signal']].tail(20)
            st.subheader("최근 데이터")
            st.dataframe(recent_data)
            txt_data = recent_data.to_csv(sep='\t')
            st.download_button(label="TXT 다운로드", data=txt_data, file_name=f"{ticker}_recent.txt", mime="text/plain")

    except Exception as e:
        st.error(f"오류 발생: {e}")
