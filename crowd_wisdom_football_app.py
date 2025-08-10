
# crowd_wisdom_football_app.py
# Streamlit app: Sabedoria das Multidões — Futebol
# Autor: ChatGPT (GPT-5 Thinking)
# Descrição: Avalia palpites de placares, calcula "sabedoria da multidão",
# odds justas 1X2 a partir das frequências e sinais de viés coletivo.

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sabedoria das Multidões — Futebol", page_icon="⚽", layout="wide")

st.title("⚽ Sabedoria das Multidões — Futebol")
st.caption("Analise palpites de placar, compare com o resultado real, estime odds justas (1X2) e detecte viés coletivo.")

with st.sidebar:
    st.header("Configurações")
    th = st.number_input("Gols do Mandante (real)", min_value=0, max_value=20, value=2, step=1)
    ta = st.number_input("Gols do Visitante (real)", min_value=0, max_value=20, value=1, step=1)
    st.markdown("---")
    input_mode = st.radio("Como deseja inserir os palpites?", ["Upload CSV", "Colar texto", "Tabela manual"], index=0)
    st.markdown("""
**Formato esperado**
- CSV: colunas `mandante, visitante` (ou `home, away`). Header opcional.
- Texto: valores separados por vírgula ou quebra de linha. Ex.: `2-1, 1-1, 0-1`.
- Tabela: edite diretamente na interface.
""")
    st.markdown("---")
    over_line = st.selectbox("Linha p/ Over/Under (total de gols)", [1.5, 2.5, 3.5, 4.5], index=1)
    show_plots = st.checkbox("Mostrar gráficos (distribuições)", value=True)

def parse_text_to_pairs(text: str):
    # Aceita "2-1", "2x1", "2 : 1", com vírgulas ou quebras de linha
    raw = [t.strip() for t in text.replace("\r", "\n").replace(",", "\n").split("\n") if t.strip()]
    pairs = []
    for item in raw:
        sep = None
        for s in ["-", "x", "X", ":", ";", " "]:  # tenta vários separadores
            if s in item:
                sep = s
                break
        if sep is None:
            # tenta split por múltiplos espaços
            parts = item.split()
        else:
            parts = [p for p in item.split(sep) if p.strip()]
        if len(parts) != 2:
            continue
        try:
            h = int(parts[0])
            a = int(parts[1])
            if h < 0 or a < 0: 
                continue
            pairs.append((h, a))
        except:
            continue
    return pairs

def load_dataframe_from_csv(uploaded_file) -> pd.DataFrame:
    # tenta ler com/sem header e com diferentes nomes de coluna
    content = uploaded_file.read()
    # tenta inferir separador
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(content), sep=sep, header=0)
        except Exception:
            continue
        if len(df.columns) >= 2:
            break
    else:
        # última tentativa sem header
        df = pd.read_csv(io.BytesIO(content), header=None)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # tenta mapear nomes
    candidates_home = [c for c in df.columns if c in ["mandante","home","casa","h","gols_mandante","gols casa","gols_home"]]
    candidates_away = [c for c in df.columns if c in ["visitante","away","fora","a","gols_visitante","gols fora","gols_away"]]
    if not candidates_home or not candidates_away:
        # se não achou, assume as duas primeiras colunas
        cols = list(df.columns)[:2]
        df = df[cols]
        df.columns = ["mandante","visitante"]
    else:
        df = df[[candidates_home[0], candidates_away[0]]]
        df.columns = ["mandante","visitante"]

    # garante inteiros não negativos
    df = df.dropna()
    df = df[(df["mandante"]>=0) & (df["visitante"]>=0)]
    df["mandante"] = df["mandante"].astype(int)
    df["visitante"] = df["visitante"].astype(int)
    return df

def outcome(h, a):
    return "H" if h > a else "A" if a > h else "D"

def evaluate_crowd(placar_real, df_palps: pd.DataFrame):
    th, ta = placar_real
    df = df_palps.copy()
    df.columns = ["mandante","visitante"]
    n = len(df)
    if n == 0:
        return None

    # métricas por palpite
    df["erro_L1"] = (df["mandante"] - th).abs() + (df["visitante"] - ta).abs()
    res_real = outcome(th, ta)
    df["outcome"] = df.apply(lambda r: outcome(r["mandante"], r["visitante"]), axis=1)
    df["acerto_resultado"] = df["outcome"] == res_real
    df["acerto_exato"] = (df["mandante"] == th) & (df["visitante"] == ta)
    df["total_gols"] = df["mandante"] + df["visitante"]

    # estatísticas do grupo
    mh, ma = df["mandante"].mean(), df["visitante"].mean()
    gh, ga = int(round(mh)), int(round(ma))
    erro_grupo = abs(gh - th) + abs(ga - ta)
    acerto_res_grupo = outcome(gh, ga) == res_real
    acerto_exato_grupo = (gh == th and ga == ta)

    # viés total (todos over/under)
    todos_over = (df["total_gols"] > (th + ta)).all()
    todos_under = (df["total_gols"] < (th + ta)).all()
    vies_total = "Over" if todos_over else "Under" if todos_under else "Não"

    # frequências 1X2 -> probabilidades -> odds justas
    freq = df["outcome"].value_counts().reindex(["H","D","A"]).fillna(0).astype(int)
    probs = freq / n
    def fair_odds(p):
        return np.where(p>0, 1.0/p, np.inf)
    odds_justas = fair_odds(probs.values)

    summary = {
        "placar_real": (th, ta),
        "media_grupo": (mh, ma),
        "placar_grupo_arred": (gh, ga),
        "erro_grupo": erro_grupo,
        "acerto_resultado_grupo": acerto_res_grupo,
        "acerto_exato_grupo": acerto_exato_grupo,
        "vies_total_gols": vies_total,
        "n_palpiteiros": n,
        "freq_1x2": {"H": int(freq["H"]), "D": int(freq["D"]), "A": int(freq["A"])},
        "prob_1x2": {"H": float(probs["H"]), "D": float(probs["D"]), "A": float(probs["A"])},
        "odds_justas_1x2": {"H": float(odds_justas[0]) if probs["H"]>0 else math.inf,
                            "D": float(odds_justas[1]) if probs["D"]>0 else math.inf,
                            "A": float(odds_justas[2]) if probs["A"]>0 else math.inf},
    }

    # ranking por erro (melhor -> pior)
    rank = df.copy()
    rank.index = np.arange(1, len(rank)+1)
    rank = rank.sort_values(["erro_L1", "acerto_exato", "acerto_resultado"], ascending=[True, False, False])

    return summary, df, rank

# ===== Entrada de dados =====
if input_mode == "Upload CSV":
    uploaded = st.file_uploader("Envie um arquivo CSV com palpites (mandante, visitante)", type=["csv"])
    if uploaded is not None:
        df_palps = load_dataframe_from_csv(uploaded)
    else:
        df_palps = pd.DataFrame(columns=["mandante","visitante"])

elif input_mode == "Colar texto":
    text = st.text_area("Cole os palpites (ex.: 2-1, 1-1, 0-1)", height=150, value="1-1, 2-0, 3-1, 0-1, 2-2, 1-0, 2-1, 1-2, 3-2, 2-1")
    pairs = parse_text_to_pairs(text)
    df_palps = pd.DataFrame(pairs, columns=["mandante","visitante"]) if pairs else pd.DataFrame(columns=["mandante","visitante"])

else:  # Tabela manual
    st.write("Edite a tabela abaixo (adicione/remova linhas conforme necessário).")
    df_default = pd.DataFrame({"mandante":[1,2,3,0,2,1,2,1,3,2],
                               "visitante":[1,0,1,1,2,0,1,2,2,1]})
    df_palps = st.data_editor(df_default, num_rows="dynamic", use_container_width=True)

# ===== Processamento =====
res = None
if len(df_palps) > 0:
    res = evaluate_crowd((th, ta), df_palps)

# ===== Saída =====
left, right = st.columns([1,1])

with left:
    st.subheader("Resumo do Grupo")
    if res is None:
        st.info("Insira palpites para ver os resultados.")
    else:
        summary, df_raw, rank = res
        st.metric("Nº de palpites", summary["n_palpiteiros"])
        st.metric("Placar real", f'{summary["placar_real"][0]} x {summary["placar_real"][1]}')
        st.metric("Média do grupo", f'{summary["media_grupo"][0]:.2f} x {summary["media_grupo"][1]:.2f}')
        st.metric("Placar do grupo (arred.)", f'{summary["placar_grupo_arred"][0]} x {summary["placar_grupo_arred"][1]}')
        st.metric("Erro do grupo (L1)", summary["erro_grupo"])
        st.metric("Acerto do resultado (grupo)", "✅" if summary["acerto_resultado_grupo"] else "❌")
        st.metric("Acerto exato (grupo)", "🎯" if summary["acerto_exato_grupo"] else "—")
        st.metric("Viés coletivo (total gols)", summary["vies_total_gols"])

with right:
    st.subheader("Odds Justas 1X2 (Frequência da Multidão)")
    if res is not None:
        freq = summary["freq_1x2"]
        prob = summary["prob_1x2"]
        odds = summary["odds_justas_1x2"]
        odds_df = pd.DataFrame({
            "Evento": ["Mandante (H)", "Empate (D)", "Visitante (A)"],
            "Frequência": [freq["H"], freq["D"], freq["A"]],
            "Probabilidade": [prob["H"], prob["D"], prob["A"]],
            "Odd Justa": [odds["H"], odds["D"], odds["A"]],
        })
        st.dataframe(odds_df, use_container_width=True)

st.markdown("---")
st.subheader("Ranking dos Apostadores (melhor → pior)")
if res is not None:
    show_cols = ["mandante","visitante","erro_L1","acerto_resultado","acerto_exato","total_gols","outcome"]
    st.dataframe(res[2][show_cols], use_container_width=True)

st.subheader("Palpites brutos")
if res is not None:
    st.dataframe(res[1], use_container_width=True)

# ===== Gráficos (matplotlib, 1 por figura, sem estilos ou cores) =====
if res is not None and show_plots:
    df = res[1]
    # Distribuição do total de gols
    fig1, ax1 = plt.subplots()
    ax1.hist(df["total_gols"], bins=range(int(df["total_gols"].min()), int(df["total_gols"].max())+2), edgecolor="black")
    ax1.set_title("Distribuição — Total de gols (palpites)")
    ax1.set_xlabel("Total de gols")
    ax1.set_ylabel("Frequência")
    st.pyplot(fig1)

    # Dispersão dos palpites (mandante vs visitante)
    fig2, ax2 = plt.subplots()
    ax2.scatter(df["mandante"], df["visitante"])
    ax2.set_title("Dispersão de palpites (mandante x visitante)")
    ax2.set_xlabel("Gols mandante (palpite)")
    ax2.set_ylabel("Gols visitante (palpite)")
    st.pyplot(fig2)

    # Over/Under em relação à linha escolhida
    over_share = (df["total_gols"] > over_line).mean() if isinstance(over_line, float) else np.nan
    fig3, ax3 = plt.subplots()
    ax3.bar(["Over", "Under"], [over_share, 1 - over_share])
    ax3.set_title(f"Proporção Over/Under vs linha {over_line}")
    ax3.set_ylabel("Proporção")
    st.pyplot(fig3)

st.markdown("---")
st.caption("Dicas: para odds comerciais, aplique margem na odd justa. Ex.: odd_com_margem = odd_justa * (1 - margem).")
