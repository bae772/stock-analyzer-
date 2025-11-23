# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="베짱이표 AI주식 분석기", layout="centered")
st.title("베짱이표 AI주식 분석기 (미국주식 AI 분석 + 투자 전략 + 에너지 패턴)")

# 사용자 입력
ticker = st.text_input("티커 입력 (예: BMR, TSLA, MARA)", value="BMR").upper()
avg_price = st.number_input("보유 평단가 입력 (없으면 0)", min_value=0.0, step=0.01, value=0.0)

if ticker:
    try:
        # 데이터 다운로드 (Open 포함)
        data = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if data.empty or len(data) < 50:
            st.error("데이터가 부족하거나 티커를 찾을 수 없습니다.")
            st.stop()

        # 필요한 컬럼 확보 (Open이 없다면 KeyError 발생하므로 안전하게 확인)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in data.columns:
                st.error(f"데이터에 '{col}' 컬럼이 없습니다.")
                st.stop()

        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        open_s = data['Open']
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']

        # ---------- 기본 스칼라 지표 (모두 .iloc[-1]로 스칼라화) ----------
        last_close = float(close.iloc[-1])
        prev_close = float(close.iloc[-2])
        change_pct = round((last_close - prev_close) / prev_close * 100, 2)
        last_date = close.index[-1].strftime('%Y-%m-%d')

        # 이동평균 & 볼린저밴드
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        last_ma20 = float(ma20.iloc[-1]) if not pd.isna(ma20.iloc[-1]) else float(close.iloc[-1])
        last_ma50 = float(ma50.iloc[-1]) if not pd.isna(ma50.iloc[-1]) else float(close.iloc[-1])

        std20 = close.rolling(20).std()
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20
        last_upper = float(upper_band.iloc[-1]) if not pd.isna(upper_band.iloc[-1]) else last_close
        last_lower = float(lower_band.iloc[-1]) if not pd.isna(lower_band.iloc[-1]) else last_close

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
        last_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

        # 거래량
        vol_ma20 = volume.rolling(20).mean()
        last_vol_ma20 = float(vol_ma20.iloc[-1]) if not pd.isna(vol_ma20.iloc[-1]) and vol_ma20.iloc[-1] != 0 else 1.0
        vol_today = float(volume.iloc[-1])
        vol_ratio = round(vol_today / last_vol_ma20, 2) if last_vol_ma20 > 0 else 1.0

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        last_macd_diff = float(macd_diff.iloc[-1]) if not pd.isna(macd_diff.iloc[-1]) else 0.0

        # ATR
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        last_atr = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else (last_close * 0.02)
        target_price = round(last_close + last_atr * 2.5, 2)
        stop_price = round(last_close - last_atr * 1.8, 2)
        target_pct = round((target_price - last_close) / last_close * 100, 1)
        stop_pct = round((stop_price - last_close) / last_close * 100, 1)

        # ---------- AI SCORE (안전한 스칼라 연산) ----------
        score = 50.0
        score += max(0, 30 - last_rsi) * 1.4
        score += change_pct * 2.0
        score += max(0, vol_ratio - 1) * 12
        score += 15 if last_close > last_ma20 else -10
        score += 10 if last_close > last_ma50 else -8
        score += 10 if last_macd_diff > 0 else -10
        score += 5 if last_close < last_lower else -5
        score = int(np.clip(score, 0, 100))

        # 등급 + 근거
        if score >= 80:
            grade, gcolor = "A (강력매수)", "#00ff00"
            reason = "RSI 저평가 + 상승추세 + MACD 양호(골든크로스) 등 복합적 긍정 신호"
        elif score >= 70:
            grade, gcolor = "A (매수)", "#33ff33"
            reason = "추세 상승 + 일부 기술지표 긍정적"
        elif score >= 60:
            grade, gcolor = "B (관망)", "#ffff33"
            reason = "단기 변동성 존재 — 신중 관망 필요"
        elif score >= 40:
            grade, gcolor = "C (주의)", "#ff9933"
            reason = "리스크 존재 — 기존 보유자 관리, 신규 진입 신중"
        else:
            grade, gcolor = "D (매도)", "#ff3333"
            reason = "하락/과열 신호 다수 — 매수 지양"

        # 평단가 대비 수익률
        if avg_price > 0:
            profit_pct = round((last_close - avg_price) / avg_price * 100, 2)
            profit_text = f"{profit_pct}% ({'수익' if profit_pct >= 0 else '손실'})"
        else:
            profit_text = "평단가 입력 없음"

        # ========== 에너지 패턴 & 잔존률 (모든 비교는 스칼라로) ==========
        # candle body : abs(close - open), shadow_total : high - low
        candle_body = (close - open_s).abs()
        shadow_total = (high - low).replace(0, np.nan)
        energy_series = (candle_body / shadow_total).replace([np.inf, -np.inf], 0).fillna(0)

        # 최근 n일 평균 대비 잔존률 (n=10)
        n_residual = 10
        energy_mean_n_series = energy_series.rolling(n_residual).mean()
        energy_mean_n_last = energy_mean_n_series.iloc[-1] if not pd.isna(energy_mean_n_series.iloc[-1]) else 0.0
        energy_last = float(energy_series.iloc[-1]) if not pd.isna(energy_series.iloc[-1]) else 0.0

        if energy_mean_n_last > 0:
            energy_residual = round(float(energy_last / energy_mean_n_last * 100), 1)
        else:
            energy_residual = round(float(energy_last * 100), 1)

        # 에너지 패턴 비교 (최근 5일 평균)
        energy_mean_5 = energy_series.rolling(5).mean().iloc[-1] if not pd.isna(energy_series.rolling(5).mean().iloc[-1]) else 0.0
        energy_pattern = "에너지 중립"
        if energy_last > energy_mean_5 * 1.05:
            energy_pattern = "상승 에너지 강화"
        elif energy_last < energy_mean_5 * 0.95:
            energy_pattern = "하락 에너지 강화"

        # 간단 전환 체크 (최근 3캔들 + 에너지 증가)
        energy_prev = float(energy_series.iloc[-2]) if len(energy_series) >= 2 and not pd.isna(energy_series.iloc[-2]) else 0.0
        if len(close) >= 3:
            if (close.iloc[-3] < close.iloc[-2] < close.iloc[-1]) and (energy_last > energy_prev):
                energy_pattern = "상승 에너지 전환"
            elif (close.iloc[-3] > close.iloc[-2] > close.iloc[-1]) and (energy_last > energy_prev):
                energy_pattern = "하락 에너지 전환"

        # ========== 복합 패턴 탐지 (스칼라 조건) ==========
        composite_pattern = "패턴 없음"

        buy_cond = (
            (last_rsi < 45)
            and (last_macd_diff > 0)
            and (last_close > last_ma20)
            and ("상승" in energy_pattern)
            and (energy_residual >= 50)
        )

        sell_cond = (
            (last_rsi > 65)
            and (last_macd_diff < 0)
            and (last_close < last_ma20)
            and ("하락" in energy_pattern)
            and (energy_residual <= 35)
        )

        if buy_cond:
            composite_pattern = "강한 복합 매수 패턴 (BUY Signal 1)"
        elif sell_cond:
            composite_pattern = "강한 복합 매도 패턴 (SELL Signal 1)"
        else:
            # 약한/보조 패턴
            partial_buy = (last_rsi < 50) and (last_macd_diff > 0)
            partial_sell = (last_rsi > 60) and (last_macd_diff < 0)
            if partial_buy and ("상승" in energy_pattern):
                composite_pattern = "약한 매수 복합 패턴 (BUY Signal 2)"
            elif partial_sell and ("하락" in energy_pattern):
                composite_pattern = "약한 매도 복합 패턴 (SELL Signal 2)"

        # ========== 단기/장기 전략 (스칼라 기반) ==========
        if last_rsi < 40 and last_macd_diff > 0 and change_pct >= -2:
            short_strategy = "단기: 소량 매수 가능 (RSI 저평가 + MACD 양호)"
        elif last_rsi > 70 or last_close > last_upper:
            short_strategy = "단기: 관망 또는 일부 매도 (과열)"
        else:
            short_strategy = "단기: 관망"

        if last_close > last_ma20 and last_ma20 > last_ma50:
            long_strategy = "장기: 상승추세, 비중 확대 가능"
        elif last_close < last_ma20 and last_ma20 < last_ma50:
            long_strategy = "장기: 하락 추세, 신규 매수 지양"
        else:
            long_strategy = "장기: 관망"

        # ========== 카드 출력 ==========
        st.markdown(f"""
        <div style="background:#000; color:white; padding:30px; border-radius:20px; 
                    text-align:center; border:3px solid #00ffcc; box-shadow:0 0 30px #00ffcc99;">
            <h1 style="color:#00ffcc; margin:0; font-size:3.2em;">{ticker}</h1>
            <h2 style="margin:8px 0; font-size:2.2em;">${last_close:.2f} ({last_date})</h2>
            <p style="color:{'#33ff33' if change_pct>=0 else '#ff3333'}; font-size:1.1em; margin:5px;">
                {'+' if change_pct>=0 else ''}{change_pct}%
            </p>

            <h3 style="color:#aaa; margin:12px 0 6px;">AI SCORE</h3>
            <h1 style="color:{'#00ff00' if score>=75 else '#ffff00' if score>=60 else '#ff9933'};
                       font-size:4.2em; margin:0; text-shadow:0 0 20px;">
                {score}
            </h1>
            <h2 style="color:{gcolor}; margin:12px 0;">등급 [{grade}]</h2>

            <p style="font-size:1.0em; margin:8px 0;">매수/매도 근거: {reason}</p>
            <p style="font-size:1.0em; margin:8px 0;">평단가 대비 수익률: {profit_text}</p>
            <p style="font-size:1.0em; margin:8px 0;">단기 전략: {short_strategy}</p>
            <p style="font-size:1.0em; margin:8px 0;">장기 전략: {long_strategy}</p>

            <p style="font-size:1.0em; margin:8px 0;">에너지 패턴: {energy_pattern}</p>
            <p style="font-size:1.0em; margin:8px 0;">에너지 잔존률: {energy_residual}%</p>
            <p style="font-size:1.05em; margin:10px 0; color:#00ffea;">복합 패턴 신호: {composite_pattern}</p>

            <div style="display:flex; justify-content:center; gap:24px; margin:18px 0;">
                <div style="background:#002200; padding:14px; border-radius:12px; min-width:140px;">
                    <div style="color:#00ff88;">TARGET</div>
                    <div style="color:#00ff00; font-size:1.2em;">${target_price}<br><small>+{target_pct}%</small></div>
                </div>
                <div style="background:#220000; padding:14px; border-radius:12px; min-width:140px;">
                    <div style="color:#ff8888;">STOP LOSS</div>
                    <div style="color:#ff3333; font-size:1.2em;">${stop_price}<br><small>{stop_pct}%</small></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ========== Plotly 차트 (Close + MA + Bollinger) ==========
        chart_len = 250
        c_chart = close[-chart_len:].dropna().reset_index(drop=True)
        ma20_chart = ma20[-chart_len:].dropna().reset_index(drop=True)
        ma50_chart = ma50[-chart_len:].dropna().reset_index(drop=True)
        upper_chart = upper_band[-chart_len:].dropna().reset_index(drop=True)
        lower_chart = lower_band[-chart_len:].dropna().reset_index(drop=True)
        x_axis = list(range(len(c_chart)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_axis, y=c_chart, mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=x_axis, y=ma20_chart, mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=x_axis, y=ma50_chart, mode='lines', name='MA50'))
        fig.add_trace(go.Scatter(x=x_axis, y=upper_chart, mode='lines', name='Upper BB', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=x_axis, y=lower_chart, mode='lines', name='Lower BB', line=dict(dash='dot')))

        # 복합 신호 마커 (현재 위치)
        marker_text = composite_pattern
        marker_color = 'green' if 'BUY' in composite_pattern else 'red' if 'SELL' in composite_pattern else 'yellow'
        fig.add_trace(go.Scatter(x=[x_axis[-1]], y=[last_close],
                                 mode='markers+text',
                                 marker=dict(color=marker_color, size=12, symbol='diamond'),
                                 text=[marker_text],
                                 textposition="top center",
                                 name='Composite Signal'))

        fig.update_layout(title=f"{ticker} 차트", xaxis_title="기간(index)", yaxis_title="가격", template='plotly_dark',
                          legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M')} │ 데이터: Yahoo Finance")

    except Exception as e:
        # 에러의 원인을 바로 보여주되, 개발 중인 메시지도 함께 제공
        st.error(f"오류 발생: {e}")
