
import streamlit as st

def implied_prob(odd: float) -> float:
    if odd is None or odd <= 0:
        return 0.0
    return 1.0 / odd

def compute_pregame_score(odd_01_open, odd_01_cur, odd_10_cur, odd_u15_cur, odd_away_open, odd_away_cur):
    prob_01 = implied_prob(odd_01_cur)
    prob_10 = implied_prob(odd_10_cur)
    prob_u15 = implied_prob(odd_u15_cur)
    prob_away_open = implied_prob(odd_away_open)
    prob_away_cur = implied_prob(odd_away_cur)

    F1 = prob_01 / prob_u15 if prob_u15 > 0 else 0.0
    F2 = odd_01_cur / odd_10_cur if odd_10_cur > 0 else 0.0
    F3 = (odd_01_cur - odd_01_open) / odd_01_open if odd_01_open > 0 else 0.0
    F4 = prob_away_cur - prob_away_open

    score = 0
    if F1 > 0.42: score += 2
    if F2 < 0.85: score += 1
    if F3 < -0.10: score += 1
    if F4 > 0.03: score += 1

    signal = "Lay 0x1" if score >= 3 else "Sem entrada"

    return {
        "score": score, "signal": signal, "F1": F1, "F2": F2, "F3": F3, "F4": F4,
        "prob_01": prob_01, "prob_10": prob_10, "prob_u15": prob_u15
    }

st.title("🎯 Score Sniper - Lay 0x1 (Versão Básica)")
st.write("Radar simples para oportunidades de Lay 0x1 baseado apenas nas odds.")

st.header("📋 Pré-Jogo")

odd_01_open = st.number_input("Odd 0x1 (Abertura)", min_value=1.01, value=7.50)
odd_01_cur = st.number_input("Odd 0x1 (Atual)", min_value=1.01, value=7.00)
odd_10_cur = st.number_input("Odd 1x0 (Atual)", min_value=1.01, value=9.00)
odd_u15_cur = st.number_input("Odd Under 1.5 (Atual)", min_value=1.01, value=2.60)
odd_away_open = st.number_input("Odd Away (Abertura)", min_value=1.01, value=3.10)
odd_away_cur = st.number_input("Odd Away (Atual)", min_value=1.01, value=2.80)

if st.button("Calcular"):
    result = compute_pregame_score(odd_01_open, odd_01_cur, odd_10_cur, odd_u15_cur, odd_away_open, odd_away_cur)

    st.subheader("Resultado")
    st.metric("Score", result["score"])
    st.metric("Sinal", result["signal"])
    st.write(f"F1 = {result['F1']:.3f}")
    st.write(f"F2 = {result['F2']:.3f}")
    st.write(f"F3 = {result['F3']:.3f}")
    st.write(f"F4 = {result['F4']:.3f}")
