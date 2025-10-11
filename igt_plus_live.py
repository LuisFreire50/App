import streamlit as st
import math

st.set_page_config(page_title="Painel IGT+ – Over Gols em Tempo Real", page_icon="⚽", layout="centered")

st.title("⚽ Painel IGT+ – Over Gols em Tempo Real")
st.write("Insira os dados atuais da partida (a cada 5 minutos) para calcular as probabilidades e odds justas dos mercados Over Gols.")

# Entradas manuais
minuto = st.slider("⏱️ Minuto atual", 0, 90, 45)
p1 = st.slider("Pressão 1 (eficiência)", 0, 50, 25)
p2 = st.slider("Pressão 2 (volume ofensivo)", 0, 50, 25)
ataques = st.slider("Ataques perigosos (últimos 10 min)", 0, 10, 3)
c1 = st.number_input("Chutes C1 (pequena área)", 0, 10, 1)
c2 = st.number_input("Chutes C2 (centro da área)", 0, 10, 1)
c3 = st.number_input("Chutes C3 (periferia da área)", 0, 10, 1)
chutes = st.slider("Chutes no gol", 0, 20, 3)
escanteios = st.slider("Escanteios", 0, 20, 3)

# Funções de cálculo
def calc_R(c1, c2, c3):
    return (c1 + 0.6*c2 + 0.3*c3) / (c1 + c2 + c3 + 1)

def calc_igt(p1, p2, c1, c2, c3, ataques):
    R = calc_R(c1, c2, c3)
    p1_adj = p1 * (1 + 0.7*R)
    return 0.5*(p1_adj/50) + 0.3*(p2/50) + 0.2*min(ataques/5, 1.2)

def goal_rate_lambda(igt):
    base = 0.028
    k = 2.2
    return max(1e-4, base * math.exp(k * (igt - 0.5)))

def prob_at_least_one_goal(t, lam):
    lamT = lam * t
    return 1 - math.exp(-lamT)

def fair_odd(p):
    return float('inf') if p <= 0 else round(1/p, 2)

# Botão de cálculo
if st.button("Calcular probabilidades"):
    igt = calc_igt(p1, p2, c1, c2, c3, ataques)
    lam = goal_rate_lambda(igt)
    tHT = max(0, 45 - minuto)
    tFT = max(0, 90 - minuto)

    pGol10 = prob_at_least_one_goal(10, lam)
    pHT = prob_at_least_one_goal(tHT, lam)
    p15FT = 1 - (math.exp(-lam*tFT)*(1 + lam*tFT))
    p25FT = 1 - (math.exp(-lam*tFT)*(1 + lam*tFT + (lam*tFT)**2/2))

    # Exibir resultados
    st.markdown("---")
    st.subheader(f"📊 Resultados – Minuto {minuto}")
    st.write(f"**IGT+:** {igt:.2f}")
    st.write(f"**Probabilidade de gol próximos 10min:** {pGol10*100:.1f}%  | Odd justa: **{fair_odd(pGol10)}**")
    st.write(f"**Over 0.5 HT:** {pHT*100:.1f}%  | Odd justa: **{fair_odd(pHT)}**")
    st.write(f"**Over 1.5 FT:** {p15FT*100:.1f}%  | Odd justa: **{fair_odd(p15FT)}**")
    st.write(f"**Over 2.5 FT:** {p25FT*100:.1f}%  | Odd justa: **{fair_odd(p25FT)}**")

    # Sinal tático
    if igt >= 0.55 and pHT >= 0.55:
        st.success("✅ Valor em Over 0.5 HT")
    elif igt >= 0.60 and p15FT >= 0.60:
        st.success("✅ Valor em Over 1.5 FT")
    elif igt >= 0.70 and pGol10 >= 0.40:
        st.warning("🔥 Valor em Over Limite (próximos 10 min)")
    else:
        st.info("⏸ Sem valor claro no momento.")


