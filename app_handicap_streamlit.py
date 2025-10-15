
import streamlit as st
import joblib
import numpy as np

# Título
st.title("🔢 Previsão de Handicap Asiático - Série A BR")

# Carregar modelo
model_bundle = joblib.load("handicap_model_bra1.joblib")
model = model_bundle['model']
le_home = model_bundle['le_home']
le_away = model_bundle['le_away']

# Input do usuário
home_team = st.selectbox("Time Mandante", sorted(le_home.classes_))
away_team = st.selectbox("Time Visitante", sorted(le_away.classes_))
home_gm_avg = st.number_input("Gols Marcados em Casa (média)", min_value=0.0, value=1.5)
home_gs_avg = st.number_input("Gols Sofridos em Casa (média)", min_value=0.0, value=1.0)
away_gm_avg = st.number_input("Gols Marcados Fora (média)", min_value=0.0, value=1.2)
away_gs_avg = st.number_input("Gols Sofridos Fora (média)", min_value=0.0, value=1.1)
line = st.selectbox("Linha de Handicap", [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
odd = st.number_input("Odd da Aposta", min_value=1.01, value=1.90)
stake = st.number_input("Stake (valor apostado)", min_value=1.0, value=100.0)

# Previsão
if st.button("📊 Analisar Aposta"):
    try:
        home_encoded = le_home.transform([home_team])[0]
        away_encoded = le_away.transform([away_team])[0]

        X_input = np.array([[home_encoded, away_encoded, home_gm_avg, home_gs_avg, away_gm_avg, away_gs_avg]])
        probs = model.predict_proba(X_input)[0]
        label_map = {0: 'Lose', 1: 'Push', 2: 'Win'}
        prob_dict = {label_map[i]: probs[i] for i in range(len(probs))}
        p_win = prob_dict.get("Win", 0)
        ev = (p_win * (odd - 1) - (1 - p_win)) * stake
        recomendacao = "✅ Apostar" if ev > 0 else "❌ Não apostar"

        st.subheader("🔍 Resultado da Análise")
        st.write("Probabilidades:", prob_dict)
        st.write("Valor Esperado (EV):", round(ev, 2))
        st.write("Recomendação:", recomendacao)
    except Exception as e:
        st.error(f"Erro: {e}")
