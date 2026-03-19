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

DB = 'https://raw.githubusercontent.com/LuisFreire50/BD-Ligas---Datasets-Completos/main/CLVU2.5.csv/db.sqlite'
MODEL_PATH = 'models/model.pkl'

# =========================
# SIDEBAR CONTROLS
# =========================
st.sidebar.header("⚙️ Configurações")

threshold = st.sidebar.slider("Probabilidade mínima (CLV)", 0.10, 0.60, 0.30)
stake_base = st.sidebar.slider("Stake base", 1.0, 10.0, 2.0)
auto_mode = st.sidebar.checkbox("🤖 Execução automática", value=False)

st.sidebar.markdown(" Painting:--- ")
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
    # In a real scenario, you would load your pre-trained model here.
    # For demonstration, we'll create a dummy model.
    # Make sure 'models/' directory exists and model.pkl is there in a real setup
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        # Create a dummy model for demonstration if not found
        class DummyModel:
            def predict_proba(self, X):
                # Return random probabilities for demonstration
                return np.random.rand(len(X), 2)
        print("Dummy model created as model.pkl not found. Please train and save a real model.")
        return DummyModel()

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

# Ensure 'models' directory exists
import os
if not os.path.exists('models'):
    os.makedirs('models')

model = load_model()

df = prepare_data(df)

features = ['odds_under_back','spread','prob']

# Handle potential missing columns for dummy model
for f in features:
    if f not in df.columns:
        df[f] = np.random.rand(len(df)) # Create dummy data if column is missing

df = df.dropna(subset=features)

# =========================
# PREDICTION
# =========================
if not df.empty:
    df['prob_CLV'] = model.predict_proba(df[features])[:,1]
else:
    st.warning("Não há dados suficientes para gerar previsões.")
    st.stop()

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

if not df.empty and 'match_id' in df.columns:
    match_id_options = df['match_id'].unique()
    if len(match_id_options) > 0:
        match_id = st.selectbox("Selecionar jogo", match_id_options)
        match_df = df[df['match_id'] == match_id]
        if not match_df.empty and 'timestamp' in match_df.columns:
            st.line_chart(match_df.set_index('timestamp')['odds_under_back'])
        else:
            st.info("Nenhum dado de odds disponível para este jogo ou coluna 'timestamp' ausente.")
    else:
        st.info("Nenhum jogo disponível para exibir o gráfico.")
else:
    st.info("Nenhum dado de jogo disponível para exibir o gráfico.")

# =========================
# MANUAL EXECUTION
# =========================
st.subheader("🎯 Execução Manual")

if st.button("Executar Top 5"):
    if not signals.empty:
        for _, row in signals.head(5).iterrows():
            bet = place_bet(row)
            st.session_state.bets.append(bet)
        st.success("Top 5 apostas executadas manualmente!")
    else:
        st.warning("Nenhum sinal disponível para execução manual.")

# =========================
# AUTO EXECUTION
# =========================
if auto_mode:
    st.warning("🤖 MODO AUTOMÁTICO ATIVO")
    if not signals.empty:
        for _, row in signals.head(5).iterrows():
            bet = place_bet(row)
            st.session_state.bets.append(bet)
        st.success("Top 5 apostas executadas automaticamente!")
    else:
        st.info("Nenhum sinal disponível para execução automática.")

# =========================
# BET HISTORY
# =========================
st.subheader("📜 Histórico de Apostas")

bets_df = pd.DataFrame(st.session_state.bets)

if not bets_df.empty:
    st.dataframe(bets_df, use_container_width=True)
else:
    st.info("Nenhuma aposta registrada ainda.")

# =========================
# RISK CONTROL (BASIC)
# =========================
st.subheader("🛡️ Controle de Risco")

max_daily_loss = st.number_input("Stop Loss Diário", value=50.0, format="%.2f")

if not bets_df.empty:
    total_stake = bets_df['stake'].sum()
    st.write(f"Total apostado: {total_stake:.2f}")
    if total_stake > max_daily_loss:
        st.error(f"ATENÇÃO: O total apostado ({total_stake:.2f}) excedeu o Stop Loss Diário ({max_daily_loss:.2f})!")
else:
    st.write("Total apostado: 0.00")

# =========================
# AUTO REFRESH
# =========================
st.caption("Atualização automática a cada 30s")
st.experimental_rerun()