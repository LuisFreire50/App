import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from datetime import datetime

# =========================
# CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("🚀 FullTrader - Under 2.5 Trading System")

DB = 'database/db.sqlite'
MODEL_PATH = 'models/model.pkl'

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Configurações")

threshold = st.sidebar.slider("Probabilidade mínima (CLV)", 0.10, 0.60, 0.30)
stake_base = st.sidebar.slider("Stake base", 1.0, 10.0, 2.0)
auto_mode = st.sidebar.checkbox("🤖 Execução automática", value=False)

st.sidebar.markdown("---")
st.sidebar.warning("Use com cautela em modo real")

# =========================
# LOAD DATA
# =========================
@st.cache_data(ttl=30)
def load_data():
    conn = sqlite3.connect(DB)
    df = pd.read_sql("SELECT * FROM odds_snapshots", conn)
    conn.close()
    return df

# =========================
# FEATURE ENGINEERING
# =========================
def prepare_data(df):
    df = df.sort_values(['match_id','timestamp'])
    
    df['spread'] = df['odds_under_lay'] - df['odds_under_back']
    df['prob'] = 1 / df['odds_under_back']
    
    df['odds_move'] = df.groupby('match_id')['odds_under_back'].diff()
    df['velocity'] = df.groupby('match_id')['odds_move'].diff()
    
    return df

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

# =========================
# BET EXECUTION (SIMULATED)
# =========================
def place_bet(row):
    return {
        "match_id": row['match_id'],
        "odds": row['odds_under_back'],
        "stake": row['stake'],
        "time": datetime.now()
    }

# =========================
# SESSION STATE
# =========================
if "bets" not in st.session_state:
    st.session_state.bets = []

# =========================
# MAIN
# =========================
df = load_data()

if df.empty:
    st.warning("Sem dados disponíveis")
    st.stop()

model = load_model()

df = prepare_data(df)

features = ['odds_under_back','spread','prob']
df = df.dropna(subset=features)

# =========================
# PREDICTION
# =========================
df['prob_CLV'] = model.predict_proba(df[features])[:,1]

# =========================
# SIGNALS
# =========================
signals = df[
    (df['prob_CLV'] > threshold) &
    (df['odds_under_back'] > 1.85) &
    (df['odds_under_back'] < 2.20)
]

signals = signals.sort_values('prob_CLV', ascending=False)

signals['stake'] = (signals['prob_CLV'] * stake_base).clip(1, 5)

# =========================
# DASHBOARD
# =========================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔥 Sinais de Aposta")
    st.dataframe(signals.head(20), use_container_width=True)

with col2:
    st.subheader("📊 Estatísticas")
    st.metric("Total Sinais", len(signals))
    st.metric("Prob média", round(signals['prob_CLV'].mean(), 3) if len(signals)>0 else 0)

# =========================
# GRAPH
# =========================
st.subheader("📈 Movimento de Odds")

match_id = st.selectbox("Selecionar jogo", df['match_id'].unique())

match_df = df[df['match_id'] == match_id]

st.line_chart(match_df.set_index('timestamp')['odds_under_back'])

# =========================
# MANUAL EXECUTION
# =========================
st.subheader("🎯 Execução Manual")

if st.button("Executar Top 5"):
    for _, row in signals.head(5).iterrows():
        bet = place_bet(row)
        st.session_state.bets.append(bet)

# =========================
# AUTO EXECUTION
# =========================
if auto_mode:
    st.warning("🤖 MODO AUTOMÁTICO ATIVO")
    
    for _, row in signals.head(5).iterrows():
        bet = place_bet(row)
        st.session_state.bets.append(bet)

# =========================
# BET HISTORY
# =========================
st.subheader("📜 Histórico de Apostas")

bets_df = pd.DataFrame(st.session_state.bets)

if not bets_df.empty:
    st.dataframe(bets_df, use_container_width=True)

# =========================
# RISK CONTROL (BASIC)
# =========================
st.subheader("🛡️ Controle de Risco")

max_daily_loss = st.number_input("Stop Loss Diário", value=50)

if not bets_df.empty:
    total_stake = bets_df['stake'].sum()
    st.write(f"Total apostado: {total_stake}")

# =========================
# AUTO REFRESH
# =========================
st.caption("Atualização automática a cada 30s")
st.experimental_rerun()
