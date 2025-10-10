
import streamlit as st
import pandas as pd
import numpy as np
import math
from typing import Optional

st.set_page_config(page_title="Score Sniper - Lay Goleada Visitante (Live)", layout="wide")

# ===============================
# Helpers
# ===============================
def tiov(ad: float, fia: float, xg_10: Optional[float]) -> float:
    """
    TIOV (Taxa de Intensidade Ofensiva do Visitante) por janela de 10'
    Fórmula base do Rafael "Score Sniper" Lima:
        TIOV_10' = (Ad * 0.6) + (FIA * 0.8) + (xG_10' * 10)
    Se xG_10' não for informado, estimamos via regressão simples:
        xG_10'_est ≈ max(0, 0.05*Ad + 0.12*FIA)   # calibragem conservadora
    """
    if xg_10 is None or (isinstance(xg_10, float) and np.isnan(xg_10)):
        xg_10_est = max(0.0, 0.05*ad + 0.12*fia)
        return ad*0.6 + fia*0.8 + xg_10_est*10.0
    return ad*0.6 + fia*0.8 + xg_10*10.0

def risk_tier(tiov_val: float) -> str:
    if tiov_val < 10:
        return "Baixo (Verde)"
    elif tiov_val < 20:
        return "Médio (Amarelo)"
    else:
        return "Alto (Vermelho)"

def expected_goals_remaining(tiov_media: float, minute: int) -> float:
    # Gols Esperados Restantes = (TIOV_médio * (90 - t)) / 40
    t_remaining = max(0, 90 - minute)
    return (tiov_media * t_remaining) / 40.0

def poisson_geq_4(lmbda: float) -> float:
    # P(X >= 4) for Poisson(lambda) = 1 - P(X <= 3)
    # P(X <= 3) = sum_{k=0..3} e^-λ λ^k / k!
    cdf_3 = sum([math.exp(-lmbda) * (lmbda**k) / math.factorial(k) for k in range(0, 4)])
    return 1.0 - cdf_3

def fair_odds_no_goleada(p_geq4: float) -> float:
    # prob de "NÃO haverá 4+ gols do visitante"
    p_no = max(1e-9, 1.0 - p_geq4)
    return 1.0 / p_no

def fair_odds_goleada(p_geq4: float) -> float:
    p_yes = max(1e-9, p_geq4)
    return 1.0 / p_yes

# ===============================
# Sidebar - Config
# ===============================
st.sidebar.header("Configuração do Painel (SofaScore)")
st.sidebar.write("Atualização a cada 10 minutos (janelas 0-10, 10-20, ..., 80-90).")

minute_now = st.sidebar.number_input("Minuto atual (0 a 90)", min_value=0, max_value=90, value=30, step=5)
poss_v_pct = st.sidebar.slider("Posse de bola do visitante (%)", 0, 100, 48)
duelos_mand_pct = st.sidebar.slider("Duelos ganhos do mandante (%)", 0, 100, 55)
saves_gk_home = st.sidebar.number_input("Defesas do goleiro (mandante) até agora", min_value=0, max_value=50, value=2)
interceptions_home = st.sidebar.number_input("Interceptações (mandante) até agora", min_value=0, max_value=50, value=6)
counters_visitor_total = st.sidebar.number_input("Contra-ataques do visitante (total)", min_value=0, max_value=100, value=4)

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros do Modelo")
st.sidebar.write("• TIOV = 0.6*Ad + 0.8*FIA + 10*xG_10'")
st.sidebar.write("• xG_10' estimado quando não houver dado: 0.05*Ad + 0.12*FIA")
st.sidebar.write("• Gols Restantes = (TIOV_médio * (90 - t)) / 40")
st.sidebar.write("• Prob(Visitante ≥4) por Poisson a partir de xG acumulado + projeção restante")

# ===============================
# Main - Entrada de Dados por Janela
# ===============================
st.title("Score Sniper • Lay Goleada Visitante (Live)")
st.caption("Modelo live com janelas de 10'. Fonte: SofaScore. Autor: Rafael \"Score Sniper\" Lima")

st.markdown("### 1) Insira os dados do visitante por janela de 10'")
st.write("Preencha **Ataques Perigosos (Ad)**, **Finalizações Dentro da Área (FIA)** e **xG_10'** (se disponível). Se não tiver xG_10', deixe em branco que o modelo estima automaticamente.")
n_windows = 9  # 0-10,...,80-90

default_rows = [{"Janela":"0-10","Ad":4,"FIA":1,"xG_10'":np.nan},
                {"Janela":"10-20","Ad":3,"FIA":1,"xG_10'":np.nan},
                {"Janela":"20-30","Ad":5,"FIA":1,"xG_10'":np.nan}]

# Build full template
template = []
labels = ["0-10","10-20","20-30","30-40","40-50","50-60","60-70","70-80","80-90"]
for i, lab in enumerate(labels):
    if i < len(default_rows):
        template.append(default_rows[i])
    else:
        template.append({"Janela": lab, "Ad": 0, "FIA": 0, "xG_10'": np.nan})

df_input = pd.DataFrame(template)
df_edit = st.data_editor(df_input, num_rows="fixed", use_container_width=True, key="editor")

# ===============================
# Cálculos por Janela
# ===============================
df_calc = df_edit.copy()
df_calc["TIOV_10'"] = df_calc.apply(lambda r: tiov(r["Ad"], r["FIA"], r["xG_10'"] if not pd.isna(r["xG_10'"]) else None), axis=1)
df_calc["xG_10_est"] = df_calc.apply(lambda r: (r["xG_10'"] if not pd.isna(r["xG_10'"]) else max(0.0, 0.05*r["Ad"] + 0.12*r["FIA"])), axis=1)
df_calc["Tier"] = df_calc["TIOV_10'"].apply(risk_tier)

# Cortar janelas futuras com base no minuto atual
def active_windows(minute_now: int):
    # 0-10 -> idx 0 ativo se minute_now > 0
    # 10-20 -> idx 1 ativo se minute_now > 10, etc.
    idx_active = int(min(minute_now // 10, 9))  # 0..9 (80-90 inclusive quando minute_now=90)
    return idx_active

idx_act = active_windows(minute_now)
df_live = df_calc.iloc[:max(1, idx_act)]  # janelas completas já passadas
xg_acum = df_live["xG_10_est"].sum()
tiov_media = df_live["TIOV_10'"].mean() if len(df_live) > 0 else 0.0

# Projeção restante de xG via TIOV_médio
gols_restantes_est = expected_goals_remaining(tiov_media if not np.isnan(tiov_media) else 0.0, minute_now)
# Converter "gols restantes esperados" em xG restante aproximado (mesma escala)
xg_restante_est = max(0.0, gols_restantes_est)  # aproximação pragmática

lambda_total = xg_acum + xg_restante_est
p_geq4 = poisson_geq_4(lambda_total)
odds_no_goleada = fair_odds_no_goleada(p_geq4)
odds_goleada = fair_odds_goleada(p_geq4)

# ===============================
# Mostrar Resultados
# ===============================
st.markdown("### 2) Resultado Live")
col1, col2, col3, col4 = st.columns(4)
col1.metric("TIOV Médio (até agora)", f"{tiov_media:.2f}")
col2.metric("xG acumulado (estimado)", f"{xg_acum:.2f}")
col3.metric("xG restante (proj.)", f"{xg_restante_est:.2f}")
col4.metric("λ Poisson (total)", f"{lambda_total:.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Prob. Visitante ≥4", f"{p_geq4*100:.2f}%")
col6.metric("Odds Justas — NÃO Goleada (Lay Goleada)", f"{odds_no_goleada:.2f}")
col7.metric("Odds Justas — Goleada (≥4)", f"{odds_goleada:.2f}")

# Sinal de decisão
if tiov_media < 10 and xg_acum < 1.0 and minute_now >= 45:
    sinal = "✅ **Sinal Verde** — Lay Goleada Visitante (EV+)"
elif tiov_media < 10 and minute_now >= 60:
    sinal = "🟨 **Sinal Amarelo** — Lay ok com stake reduzida"
elif tiov_media >= 20 or p_geq4 > 0.18:
    sinal = "🛑 **Sinal Vermelho** — Evitar Lay (ritmo/risco alto)"
else:
    sinal = "ℹ️ **Neutro** — Aguardando confirmação do ritmo"

st.markdown(f"### 3) Decisão\n{sinal}")

# ===============================
# Tabela por Janela + Destaques
# ===============================
st.markdown("### 4) Detalhe por janela (10')")
def highlight_tiers(row):
    color = ""
    if row["Tier"].startswith("Baixo"):
        color = "background-color: #d4edda"  # verde claro
    elif row["Tier"].startswith("Médio"):
        color = "background-color: #fff3cd"  # amarelo claro
    else:
        color = "background-color: #f8d7da"  # vermelho claro
    return [color]*len(row)

st.dataframe(df_calc.style.apply(highlight_tiers, axis=1), use_container_width=True)

# ===============================
# Dicas operacionais (lado direito)
# ===============================
with st.expander("🔧 Regras práticas — SofaScore (atualização 10')"):
    st.markdown("""
**Parâmetros-chave para Lay Goleada Visitante (seguro):**
- Ataques perigosos do visitante: **≤ 4** por 10'
- Finalizações dentro da área: **≤ 1.5** por 10'
- xG acumulado do visitante: **< 0.8** até os **60'**
- Duelos ganhos do mandante: **≥ 55%**
- Interceptações do mandante: **≥ 8** até os **60'**
- Posse visitante: **< 55%** (visitante sem controle territorial)

**Alertas (evitar Lay):**
- TIOV_10' ≥ **20** em 2+ janelas consecutivas
- xG acumulado ≥ **1.5** até os **60'**
- Poss. visitante ≥ **65%**
- Contra-ataques do visitante > **4** por 10' em qualquer janela
""")

with st.expander("📐 Notas do Modelo"):
    st.markdown("""
- **TIOV** resume ritmo ofensivo: `0.6*Ad + 0.8*FIA + 10*xG_10'`.
- Se **xG_10'** não estiver disponível no SofaScore naquela liga, **estimamos** via `0.05*Ad + 0.12*FIA` (conservador).
- Projeção de gols restantes: `(TIOV_médio * (90 - t)) / 40`.
- **Poisson**: usamos `λ = xG_acum + xG_restante_est` para estimar `P(Visitante ≥4)` e odds justas.
- O modelo é **live-first**: serve para **confirmar** entrada de Lay com disciplina de banca.
""")

st.success("Pronto. Preencha as janelas a cada 10' e acompanhe o sinal.")
