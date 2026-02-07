#app.py
#Streamlit app: Gatilhos Over (10 sinais)
#Rode com: streamlit run app.py


import streamlit as st

st.set_page_config(page_title="Over Gols – 10 Gatilhos (Live)", layout="wide")

st.title("📈 Over Gols (Live) – 10 Gatilhos Objetivos")
st.caption("Preencha os dados do Live Scanner e o app calcula se existe cenário estatístico para Over.")

# -----------------------------
# Helpers
# -----------------------------
def xg_expected_by_minute(minute: int) -> float:
    """
    Linha-base (cola mental):
    20' -> 0.60 | 30' -> 1.10 | 45' -> 1.60 | 60' -> 2.30
    Interpolação linear entre pontos.
    """
    points = [(0, 0.0), (20, 0.60), (30, 1.10), (45, 1.60), (60, 2.30), (90, 3.2)]
    minute = max(0, min(90, minute))

    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= minute <= x1:
            if x1 == x0:
                return y0
            t = (minute - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return points[-1][1]


def fmt_bool(ok: bool) -> str:
    return "✅ OK" if ok else "❌ Falhou"


# -----------------------------
# Inputs
# -----------------------------
colA, colB = st.columns([1.2, 1])

with colA:
    st.subheader("🧾 Entradas do Live Scanner")

    minute = st.slider("Minuto atual", 0, 90, 30, 1)

    st.markdown("### xG / Qualidade")
    xg_home = st.number_input("xG Casa", min_value=0.0, value=0.55, step=0.01, format="%.2f")
    xg_away = st.number_input("xG Fora", min_value=0.0, value=0.60, step=0.01, format="%.2f")
    xgot_total = st.number_input("xGoT TOTAL (se tiver)", min_value=0.0, value=0.85, step=0.01, format="%.2f")

    xgop_total = st.number_input("xG Bola Rolando (xGOP) TOTAL", min_value=0.0, value=0.95, step=0.01, format="%.2f")
    npxg_total = st.number_input("npxG TOTAL (xG sem pênaltis)", min_value=0.0, value=1.05, step=0.01, format="%.2f")
    xgsp_total = st.number_input("xG Bola Parada TOTAL (xGSP) (opcional)", min_value=0.0, value=0.15, step=0.01, format="%.2f")
    xgc_total = st.number_input("xG de Escanteios TOTAL (xGC) (opcional)", min_value=0.0, value=0.08, step=0.01, format="%.2f")

    st.markdown("### Volume / Pressão")
    shots_in_box = st.number_input("Finalizações dentro da área (TOTAL)", min_value=0, value=3, step=1)
    shots_on_target = st.number_input("Finalizações no gol (TOTAL)", min_value=0, value=2, step=1)
    dangerous_attacks = st.number_input("Ataques perigosos (TOTAL)", min_value=0, value=14, step=1)
    key_passes = st.number_input("Passes-chave (TOTAL)", min_value=0, value=3, step=1)

    st.markdown("### Contexto (defesa frágil)")
    xga_total = st.number_input("xGA TOTAL (xG sofrido total do jogo) (opcional)", min_value=0.0, value=1.30, step=0.01, format="%.2f")

    st.markdown("--- Expansion")
    st.markdown("### Ajustes")
    st.write("Você pode ajustar os limiares conforme seu perfil.")
    xg_edge_pct = st.slider("Margem acima do esperado (xG) para considerar edge", 0, 60, 30, 5)

with colB:
    st.subheader("⚙️ Limiar (padrão recomendado)")
    # thresholds dependem do tempo; vamos usar as regras simples:
    # In-box: 30' ≥3 | 45' ≥5 | 60' ≥7 (interpolado)
    # SOT: 30' ≥2 | 45' ≥3
    # DA: 30' ≥12 | 45' ≥20
    # KP: 30' ≥3 | 45' ≥5
    # Ratios: xGoT/xG ≥ 0.70 | xGOP/xG ≥ 0.70
    # npxG close: abs(xG - npxG) pequeno (<=0.25) — sinal de não depender de pênalti
    # Set piece context: xGSP + xGC >= 0.25 (sinal extra) — opcional
    # xGA: >= 1.30 até 60' (sinal extra)

    st.info(
        "Os 10 gatilhos abaixo são calculados automaticamente com base no minuto.\n\n"
        "Dica: Para entrada manual, eu geralmente exijo **pelo menos 7/10** (com xG-edge + área + SOT)."
    )

# -----------------------------
# Compute
# -----------------------------
xg_total = xg_home + xg_away
xg_expected = xg_expected_by_minute(minute)
xg_edge_needed = xg_expected * (1 + xg_edge_pct / 100)

# thresholds by minute (simple piecewise + interpolation)
def interpolate_threshold(minute: int, p1, p2, p3):
    # p1 at 30, p2 at 45, p3 at 60
    if minute <= 30:
        return p1
    if 30 < minute <= 45:
        return p1 + (p2 - p1) * ((minute - 30) / 15)
    if 45 < minute <= 60:
        return p2 + (p3 - p2) * ((minute - 45) / 15)
    # after 60, keep p3
    return p3

inbox_thr = round(interpolate_threshold(minute, 3, 5, 7))
sot_thr = round(interpolate_threshold(minute, 2, 3, 4))  # após 60, exigir 4
da_thr = round(interpolate_threshold(minute, 12, 20, 28))
kp_thr = round(interpolate_threshold(minute, 3, 5, 7))

# 10 gatilhos
# 1) xG acima do esperado (edge)
t1 = xg_total >= xg_edge_needed

# 2) Finalizações dentro da área
t2 = shots_in_box >= inbox_thr

# 3) Finalizações no gol
t3 = shots_on_target >= sot_thr

# 4) xGoT ratio
t4 = (xg_total > 0) and (xgot_total >= 0.70 * xg_total)

# 5) Ataques Perigosos
t5 = dangerous_attacks >= da_thr

# 6) Passes-chave
t6 = key_passes >= kp_thr

# 7) xGOP ratio
t7 = (xg_total > 0) and (xgop_total >= 0.70 * xg_total)

# 8) npxG “próximo” do xG total (não depender de pênalti)
t8 = abs(xg_total - npxg_total) <= 0.25

# 9) Bola parada/escanteios dando suporte (sinal extra)
# (não obrigatório em todo jogo, mas útil quando está alto)
t9 = (xgsp_total + xgc_total) >= 0.25

# 10) Defesa “cede” (xGA alto)
t10 = xga_total >= 1.30 if minute >= 30 else xga_total >= 0.90

triggers = [
    ("xG total ≥ xG esperado + margem", t1, f"xG={xg_total:.2f} | Esperado={xg_expected:.2f} | Limiar={xg_edge_needed:.2f}"),
    ("Finalizações dentro da área ≥ limiar", t2, f"Área={shots_in_box} | Limiar={inbox_thr}"),
    ("Finalizações no gol ≥ limiar", t3, f"No gol={shots_on_target} | Limiar={sot_thr}"),
    ("xGoT ≥ 70% do xG", t4, f"xGoT={xgot_total:.2f} | 70% xG={0.70*xg_total:.2f}"),
    ("Ataques perigosos ≥ limiar", t5, f"DA={dangerous_attacks} | Limiar={da_thr}"),
    ("Passes-chave ≥ limiar", t6, f"KP={key_passes} | Limiar={kp_thr}"),
    ("xG bola rolando (xGOP) ≥ 70% do xG", t7, f"xGOP={xgop_total:.2f} | 70% xG={0.70*xg_total:.2f}"),
    ("npxG próximo do xG (sem depender de pênalti)", t8, f"npxG={npxg_total:.2f} | |xG-npxG|={abs(xg_total-npxg_total):.2f} (≤0.25)"),
    ("Bola parada + escanteios (xGSP+xGC) ≥ 0.25", t9, f"xGSP+xGC={(xgsp_total+xgc_total):.2f} (≥0.25)"),
    ("xGA alto (defesa cedendo) ≥ limiar", t10, f"xGA={xga_total:.2f} | Limiar={'0.90' if minute<30 else '1.30'}"),
]

score = sum(1 for _, ok, _ in triggers if ok)

# -----------------------------
# Output
# -----------------------------
st.markdown("--- Expansion")
left, right = st.columns([1.2, 1])

with left:
    st.subheader("✅ Resultado dos 10 gatilhos")
    for name, ok, detail in triggers:
        st.write(f"**{fmt_bool(ok)} — {name}**")
        st.caption(detail)

with right:
    st.subheader("📌 Decisão")
    st.metric("Score", f"{score}/10")

    # Recomendação (simples e prática)
    # Regra: 7+ = cenário forte; 6 = cenário ok se odds boas e jogo vivo; <6 = evitar
    if score >= 7 and t1 and (t2 or t3):
        st.success("Cenário FORTE para Over (edge estatístico presente).")
        st.write("Sugestão manual: procurar **Over Asiático** (2.0/2.25/1.75 conforme linha) com odds ainda não ajustadas.")
    elif score == 6 and t1 and (t2 or t3):
        st.warning("Cenário BOM, mas exige preço (odds) favorável e jogo com ritmo contínuo.")
    else:
        st.error("Sem edge suficiente. Melhor **não entrar** em Over neste momento.")

    st.markdown("### Dicas rápidas (manual)")
    st.write("- Priorize: **xG-edge + Área + No gol**.")
    st.write("- Evite: xG inflado por evento isolado (pênalti/chute único).")
    st.write("- Se o score cair (ritmo morre), não force entrada.")

st.caption("Obs.: limiares são heurísticos e foram desenhados para operação manual em live. Ajuste conforme seu histórico/ligas.")
