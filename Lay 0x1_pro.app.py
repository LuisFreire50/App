import sys
if 'google.colab' in sys.modules:
  !pip install streamlit
import streamlit as st
import pandas as pd
from datetime import datetime

def implied_prob(odd: float) -> float:
    try:
        odd = float(odd)
    except Exception:
        return 0.0
    return 0.0 if odd <= 0 else 1.0 / odd

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def pregame_features(odd_01_open, odd_01_cur, odd_10_cur, odd_u15_cur, odd_away_open, odd_away_cur):
    prob_01 = implied_prob(odd_01_cur)
    prob_u15 = implied_prob(odd_u15_cur)
    prob_away_open = implied_prob(odd_away_open)
    prob_away_cur = implied_prob(odd_away_cur)

    F1 = (prob_01 / prob_u15) if prob_u15 > 0 else 0.0
    F2 = (odd_01_cur / odd_10_cur) if odd_10_cur and odd_10_cur > 0 else 0.0
    F3 = ((odd_01_cur - odd_01_open) / odd_01_open) if odd_01_open and odd_01_open > 0 else 0.0
    F4 = prob_away_cur - prob_away_open

    score = 0
    if F1 > 0.42: score += 2
    if F2 < 0.85: score += 1
    if F3 < -0.10: score += 1
    if F4 > 0.03: score += 1

    signal = "Lay 0x1" if score >= 3 else "Sem entrada"
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, score=score, signal=signal)

def live_features(minute, odd_01_open, odd_01_live, odd_10_live, odd_u15_live, odd_u25_live, odd_away_live):
    minute = int(minute)
    prob_01 = implied_prob(odd_01_live)
    prob_u15 = implied_prob(odd_u15_live)
    prob_u25 = implied_prob(odd_u25_live)
    prob_away_live = implied_prob(odd_away_live)

    F1 = (prob_01 / prob_u15) if prob_u15 > 0 else 0.0
    F2 = (odd_01_live / odd_10_live) if odd_10_live and odd_10_live > 0 else 0.0
    F3 = ((odd_01_live - odd_01_open) / odd_01_open) if odd_01_open and odd_01_open > 0 else 0.0
    F4 = prob_away_live - prob_u25  # Away vs Under2.5

    if minute <= 20: f1_ok = F1 > 0.50
    elif minute <= 40: f1_ok = F1 > 0.55
    elif minute <= 60: f1_ok = F1 > 0.60
    else: f1_ok = F1 > 0.65

    score = 0
    if f1_ok: score += 2
    if F2 < 0.80: score += 1
    if F3 < -0.10: score += 1
    if F4 > 0.05: score += 1

    signal = "Lay 0x1" if score >= 3 else "Sem entrada"
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, score=score, signal=signal)

def init_state():
    if "log" not in st.session_state:
        st.session_state.log = pd.DataFrame(columns=[
            "timestamp","mode","match","minute","odd_0x1","liability","stake",
            "score","signal","F1","F2","F3","F4","notes"
        ])

def add_log(row):
    st.session_state.log = pd.concat([st.session_state.log, pd.DataFrame([row])], ignore_index=True)

st.set_page_config(page_title="Score Sniper - Lay 0x1 PRO", layout="wide")
init_state()

st.title("🎯 Score Sniper — Lay 0x1 (PRO)")
st.caption("Pré-jogo + Live + Batch + Log + Export + Stake (tudo odds-only).")

t1, t2, t3, t4 = st.tabs(["📋 Pré-jogo", "📡 Live", "🗂️ Batch", "📒 Log"])

with t1:
    st.subheader("Pré-jogo")
    c1, c2, c3 = st.columns(3)
    with c1:
        match = st.text_input("Jogo", "Mandante vs Visitante")
        odd_01_open = st.number_input("Odd 0x1 (Abertura)", 1.01, 1000.0, 7.50, 0.01)
        odd_01_cur = st.number_input("Odd 0x1 (Atual)", 1.01, 1000.0, 7.00, 0.01)
    with c2:
        odd_10_cur = st.number_input("Odd 1x0 (Atual)", 1.01, 1000.0, 9.00, 0.01)
        odd_u15_cur = st.number_input("Odd Under 1.5 (Atual)", 1.01, 1000.0, 2.60, 0.01)
    with c3:
        odd_away_open = st.number_input("Odd Away (Abertura)", 1.01, 1000.0, 3.10, 0.01)
        odd_away_cur = st.number_input("Odd Away (Atual)", 1.01, 1000.0, 2.80, 0.01)
        notes = st.text_input("Notas", "")

    r = pregame_features(odd_01_open, odd_01_cur, odd_10_cur, odd_u15_cur, odd_away_open, odd_away_cur)
    a,b,c,d = st.columns(4)
    a.metric("Score", r["score"])
    b.metric("Sinal", r["signal"])
    c.metric("F1", f"{r['F1']:.3f}")
    d.metric("F2", f"{r['F2']:.3f}")

    st.markdown("---")
    st.subheader("Stake (Lay)")
    c1, c2, c3 = st.columns(3)
    with c1:
        bankroll = st.number_input("Banca (R$)", 0.0, 1e12, 10000.0, 100.0)
        risk_pct = st.number_input("Risco (% banca)", 0.01, 5.0, 0.50, 0.05)
    with c2:
        lay_odd = st.number_input("Odd Lay 0x1", 1.01, 1000.0, float(odd_01_cur), 0.01)
        liability = bankroll * (risk_pct/100.0)
        st.write(f"Responsabilidade-alvo: **R$ {liability:,.2f}**")
    with c3:
        stake = liability / (lay_odd - 1.0) if lay_odd > 1 else 0.0
        st.metric("Stake sugerida", f"R$ {stake:,.2f}")

    if st.button("Salvar no log (Pré-jogo)"):
        add_log(dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mode="pregame",
            match=match,
            minute=None,
            odd_0x1=float(odd_01_cur),
            liability=float(liability),
            stake=float(stake),
            score=int(r["score"]),
            signal=r["signal"],
            F1=float(r["F1"]),
            F2=float(r["F2"]),
            F3=float(r["F3"]),
            F4=float(r["F4"]),
            notes=notes
        ))
        st.success("Salvo no log.")

with t2:
    st.subheader("Live")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        match_l = st.text_input("Jogo (Live)", "Mandante vs Visitante", key="ml")
        minute = st.number_input("Minuto", 1, 120, 35, 1)
        scoreline = st.selectbox("Placar (informativo)", ["0-0","0-1","1-0","1-1","Outro"], 0)
    with c2:
        odd_01_open_l = st.number_input("Odd 0x1 (Abertura)", 1.01, 1000.0, 7.50, 0.01, key="o1")
        odd_01_live = st.number_input("Odd 0x1 (Live)", 1.01, 1000.0, 4.50, 0.01, key="o2")
        odd_10_live = st.number_input("Odd 1x0 (Live)", 1.01, 1000.0, 6.50, 0.01, key="o3")
    with c3:
        odd_u15_live = st.number_input("Odd Under 1.5 (Live)", 1.01, 1000.0, 2.10, 0.01, key="o4")
        odd_u25_live = st.number_input("Odd Under 2.5 (Live)", 1.01, 1000.0, 1.60, 0.01, key="o5")
    with c4:
        odd_away_live = st.number_input("Odd Away (Live)", 1.01, 1000.0, 2.30, 0.01, key="o6")
        notes_l = st.text_input("Notas", "", key="nl")

    rl = live_features(minute, odd_01_open_l, odd_01_live, odd_10_live, odd_u15_live, odd_u25_live, odd_away_live)
    a,b,c,d = st.columns(4)
    a.metric("Score", rl["score"])
    b.metric("Sinal", rl["signal"])
    c.metric("F1", f"{rl['F1']:.3f}")
    d.metric("F4", f"{rl['F4']:.3f}")

    st.markdown("---")
    st.subheader("Stake (Lay — Live)")
    c1, c2, c3 = st.columns(3)
    with c1:
        bankroll = st.number_input("Banca (R$)", 0.0, 1e12, 10000.0, 100.0, key="bl")
        risk_pct = st.number_input("Risco (% banca)", 0.01, 5.0, 0.50, 0.05, key="rpl")
    with c2:
        lay_odd = st.number_input("Odd Lay 0x1", 1.01, 1000.0, float(odd_01_live), 0.01, key="lol")
        liability = bankroll * (risk_pct/100.0)
        st.write(f"Responsabilidade-alvo: **R$ {liability:,.2f}**")
    with c3:
        stake = liability / (lay_odd - 1.0) if lay_odd > 1 else 0.0
        st.metric("Stake sugerida", f"R$ {stake:,.2f}")

    if st.button("Salvar no log (Live)"):
        add_log(dict(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            mode="live",
            match=match_l,
            minute=int(minute),
            odd_0x1=float(odd_01_live),
            liability=float(liability),
            stake=float(stake),
            score=int(rl["score"]),
            signal=rl["signal"],
            F1=float(rl["F1"]),
            F2=float(rl["F2"]),
            F3=float(rl["F3"]),
            F4=float(rl["F4"]),
            notes=f"placar={scoreline}; {notes_l}".strip()
        ))
        st.success("Salvo no log.")

with t3:
    st.subheader("Batch (CSV/XLSX)")
    mode = st.selectbox("Modo", ["pregame","live"], 0)
    upl = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
    st.caption("Pré-jogo: match, odd_01_open, odd_01_cur, odd_10_cur, odd_u15_cur, odd_away_open, odd_away_cur")
    st.caption("Live: match, minute, odd_01_open, odd_01_live, odd_10_live, odd_u15_live, odd_u25_live, odd_away_live")

    if upl is not None:
        df = pd.read_csv(upl) if upl.name.lower().endswith(".csv") else pd.read_excel(upl)
        st.dataframe(df.head(30), use_container_width=True)
        out, errs = [], 0

        if mode == "pregame":
            req = ["match","odd_01_open","odd_01_cur","odd_10_cur","odd_u15_cur","odd_away_open","odd_away_cur"]
            miss = [c for c in req if c not in df.columns]
            if miss:
                st.error(f"Faltando colunas: {miss}")
            else:
                for _, r in df.iterrows():
                    try:
                        rr = pregame_features(safe_float(r["odd_01_open"]),
                                              safe_float(r["odd_01_cur"]),
                                              safe_float(r["odd_10_cur"]),
                                              safe_float(r["odd_u15_cur"]),
                                              safe_float(r["odd_away_open"]),
                                              safe_float(r["odd_away_cur"]))
                        row = dict(r); row.update(rr)
                        out.append(row)
                    except Exception:
                        errs += 1
        else:
            req = ["match","minute","odd_01_open","odd_01_live","odd_10_live","odd_u15_live","odd_u25_live","odd_away_live"]
            miss = [c for c in req if c not in df.columns]
            if miss:
                st.error(f"Faltando colunas: {miss}")
            else:
                for _, r in df.iterrows():
                    try:
                        rr = live_features(int(r["minute"]),
                                           safe_float(r["odd_01_open"]),
                                           safe_float(r["odd_01_live"]),
                                           safe_float(r["odd_10_live"]),
                                           safe_float(r["odd_u15_live"]),
                                           safe_float(r["odd_u25_live"]),
                                           safe_float(r["odd_away_live"]))
                        row = dict(r); row.update(rr)
                        out.append(row)
                    except Exception:
                        errs += 1

        if out:
            out_df = pd.DataFrame(out)
            st.success(f"Processado: {len(out_df)} linhas. Erros: {errs}.")
            st.dataframe(out_df, use_container_width=True)
            st.download_button("Baixar resultado (CSV)", data=out_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"batch_{mode}_result.csv", mime="text/csv")

with t4:
    st.subheader("Log & Export")
    if st.session_state.log.empty:
        st.info("Log vazio.")
    else:
        df = st.session_state.log.copy()
        c1, c2, c3 = st.columns(3)
        with c1:
            mf = st.selectbox("Modo", ["todos","pregame","live"], 0)
        with c2:
            min_score = st.number_input("Score mínimo", 0, 10, 0, 1)
        with c3:
            q = st.text_input("Buscar", "")

        if mf != "todos":
            df = df[df["mode"] == mf]
        df = df[df["score"] >= int(min_score)]
        if q.strip():
            df = df[df["match"].fillna("").astype(str).str.contains(q, case=False, na=False)]

        st.dataframe(df, use_container_width=True)

        a,b,c,d = st.columns(4)
        a.metric("Entradas", len(df))
        b.metric("Lay 0x1", int((df["signal"]=="Lay 0x1").sum()))
        c.metric("Stake total (R$)", f"{df['stake'].fillna(0).sum():,.2f}")
        d.metric("Resp. total (R$)", f"{df['liability'].fillna(0).sum():,.2f}")

        st.line_chart(df["score"].fillna(0), height=220)
        st.line_chart(df["odd_0x1"].fillna(0), height=220)

        st.download_button("Baixar LOG (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="lay01_log.csv", mime="text/csv")

        if st.button("Limpar log"):
            st.session_state.log = st.session_state.log.iloc[0:0]
            st.success("Log limpo.")
