
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression

# Simular dados de treinamento (modelo foi treinado com isso anteriormente)
def treinar_modelo():
    np.random.seed(42)
    n_samples = 500
    media_gols_casa = np.random.normal(1.5, 0.5, n_samples)
    media_gols_fora = np.random.normal(1.2, 0.4, n_samples)
    chutes_no_alvo = np.random.normal(10, 3, n_samples)
    escanteios = np.random.normal(5, 2, n_samples)
    prob_over = 1 / (1 + np.exp(-1 * (media_gols_casa + media_gols_fora - 2.5)))
    over_2_5 = np.random.binomial(1, prob_over)

    X = np.column_stack((media_gols_casa, media_gols_fora, chutes_no_alvo, escanteios))
    y = over_2_5
    model = LogisticRegression()
    model.fit(X, y)
    return model

# Treinar o modelo uma vez
model = treinar_modelo()

# Interface do Streamlit
st.title("Avaliador de Aposta: Over 2.5 Gols")

# Entradas do usuário
st.header("📥 Insira as estatísticas da partida")

media_gols_casa = st.number_input("Média de gols do time da casa", min_value=0.0, max_value=5.0, step=0.1)
media_gols_fora = st.number_input("Média de gols do time visitante", min_value=0.0, max_value=5.0, step=0.1)
chutes_no_alvo = st.number_input("Chutes no alvo (total)", min_value=0.0, max_value=30.0, step=0.5)
escanteios = st.number_input("Número de escanteios (total)", min_value=0.0, max_value=20.0, step=0.5)

st.header("💰 Insira as odds do mercado")

odds_over = st.number_input("Odds para Over 2.5", min_value=1.01, max_value=10.0, step=0.01)

if st.button("🔍 Avaliar Aposta"):
    # Previsão
    X_input = np.array([[media_gols_casa, media_gols_fora, chutes_no_alvo, escanteios]])
    prob = model.predict_proba(X_input)[0][1]

    # Cálculo do valor esperado
    valor_esperado = (prob * (odds_over - 1)) - (1 - prob)
    apostar = valor_esperado > 0

    # Resultados
    st.subheader("📊 Resultado da Análise")
    st.write(f"Probabilidade estimada de Over 2.5 gols: **{prob:.2%}**")
    st.write(f"Valor Esperado (EV): **{valor_esperado:.3f}**")
    
    if apostar:
        st.success("✅ Aposta com valor esperado positivo! Considere apostar.")
    else:
        st.error("❌ Aposta com valor esperado negativo. Evite apostar.")
