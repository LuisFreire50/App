#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Closing Line Model (nível "sindicato") — 1x2 + O/U
Autor: ChatGPT (customizado para Fulltrader)

Objetivo:
- Usar APENAS Closing Line (odds 1x2 + odds O/U) para:
  1) Remover margem do book (probabilidades "fair")
  2) Calibrar o mercado (logistic / multinomial logistic)
  3) Aplicar shrinkage para o mercado (alpha otimizado)
  4) Ajuste residual com KNN kernel (opcional)
  5) Walk-forward backtest (expanding window)
  6) Relatórios: LogLoss/Brier, calibração por bins, e simulação de apostas (ROI, DD)
  7) Exportar previsões + apostas sugeridas

Uso (Colab / local):
python hybrid_closing_sindicato.py \
  --data-url "https://raw.githubusercontent.com/USER/REPO/main/data/matches.csv" \
  --date-col "Date" \
  --outdir "out" \
  --min-edge 0.02 \
  --stake-mode "fkelly" \
  --fkelly 0.25

Observações:
- Para CLV de verdade você precisaria também da odd "que você pegou" no momento da aposta.
  Com apenas Closing Line, CLV não se aplica; aqui reportamos "edge vs closing" e performance simulada apostando no closing.
"""

import argparse
import io
import os
import sys
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ----------------------------
# Utilidades de probabilidades
# ----------------------------
def _safe_div(a, b, eps=1e-12):
    return a / (b + eps)

def odds_to_implied_probs_1x2(odds_h, odds_d, odds_a):
    """Probabilidades implícitas (com margem)."""
    inv = np.vstack([_safe_div(1.0, odds_h), _safe_div(1.0, odds_d), _safe_div(1.0, odds_a)]).T
    return inv

def remove_margin_proportional(inv_probs):
    """Remove margem via normalização proporcional: p_fair = inv / sum(inv)."""
    s = inv_probs.sum(axis=1, keepdims=True)
    return _safe_div(inv_probs, s)

def odds_to_fair_probs_1x2(odds_h, odds_d, odds_a):
    inv = odds_to_implied_probs_1x2(odds_h, odds_d, odds_a)
    return remove_margin_proportional(inv)

def odds_to_implied_prob_ou(odds_over, odds_under):
    inv_o = _safe_div(1.0, odds_over)
    inv_u = _safe_div(1.0, odds_under)
    return np.vstack([inv_o, inv_u]).T

def odds_to_fair_prob_over(odds_over, odds_under):
    inv = odds_to_implied_prob_ou(odds_over, odds_under)
    fair = remove_margin_proportional(inv)
    return fair[:, 0]  # prob Over fair

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return _safe_div(e, e.sum(axis=1, keepdims=True))

# ----------------------------
# Features derivadas do mercado
# ----------------------------
def market_features(p1x2_fair: np.ndarray, pover_fair: np.ndarray) -> pd.DataFrame:
    """
    p1x2_fair: (n,3) => [pH,pD,pA]
    pover_fair: (n,)
    """
    pH, pD, pA = p1x2_fair[:, 0], p1x2_fair[:, 1], p1x2_fair[:, 2]
    fav = np.maximum(pH, pA)
    dog = np.minimum(pH, pA)
    diff = fav - dog
    # entropia (incerteza)
    ent = -(pH*np.log(np.clip(pH,1e-12,1)) + pD*np.log(np.clip(pD,1e-12,1)) + pA*np.log(np.clip(pA,1e-12,1)))
    # draw relative
    draw_skew = pD - (pH + pA)/2.0
    # shape 1x2
    balance = np.abs(pH - pA)
    # total-goals proxy (simples): usar pover e interação com favorito
    pover = pover_fair

    df = pd.DataFrame({
        "pH": pH, "pD": pD, "pA": pA,
        "fav": fav, "dog": dog,
        "fav_minus_dog": diff,
        "entropy_1x2": ent,
        "draw_skew": draw_skew,
        "balance_HA": balance,
        "pOver": pover,
        "pOver_x_fav": pover * fav,
        "pOver_x_entropy": pover * ent,
        "fav_x_entropy": fav * ent,
    })
    return df

# ----------------------------
# Métricas e calibração
# ----------------------------
def brier_multiclass(y_true: np.ndarray, p_pred: np.ndarray, n_classes: int = 3) -> float:
    y_onehot = np.zeros((len(y_true), n_classes), dtype=float)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((p_pred - y_onehot) ** 2, axis=1)))

def brier_binary(y_true01: np.ndarray, p_pred: np.ndarray) -> float:
    y_true01 = y_true01.astype(float)
    return float(np.mean((p_pred - y_true01) ** 2))

def calibration_bins(y_true01: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            rows.append({"bin": b, "count": 0, "p_mean": np.nan, "y_rate": np.nan})
        else:
            rows.append({"bin": b, "count": int(m.sum()), "p_mean": float(p_pred[m].mean()), "y_rate": float(y_true01[m].mean())})
    return pd.DataFrame(rows)

# ----------------------------
# Simulação de apostas
# ----------------------------
@dataclass
class Bet:
    idx: int
    market: str        # "1x2" ou "OU"
    selection: str     # "H","D","A" ou "Over","Under"
    prob_model: float
    prob_market: float
    odds: float
    edge: float        # prob_model - prob_market
    stake: float
    won: int           # 1 ou 0
    pnl: float

def kelly_fraction(p: float, o: float) -> float:
    """Kelly para odds decimais. Retorna fração ótima do bankroll (p*(o-1)-(1-p))/ (o-1)"""
    b = o - 1.0
    q = 1.0 - p
    f = (p*b - q) / (b + 1e-12)
    return max(0.0, f)

def simulate_bets(df_test: pd.DataFrame,
                  p_market_1x2: np.ndarray,
                  p_model_1x2: np.ndarray,
                  odds_1x2: np.ndarray,
                  y_1x2: np.ndarray,
                  p_market_over: np.ndarray,
                  p_model_over: np.ndarray,
                  odds_over: np.ndarray,
                  odds_under: np.ndarray,
                  y_over01: np.ndarray,
                  min_edge: float = 0.02,
                  stake_mode: str = "flat",
                  flat_stake: float = 1.0,
                  fkelly: float = 0.25,
                  max_kelly: float = 0.03,
                  max_bets_per_game: int = 1,
                  bankroll0: float = 100.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simula apostas no closing (teórico).
    - Escolhe a maior edge por jogo, respeitando max_bets_per_game.
    - stake_mode: flat | fkelly
    """
    bankroll = bankroll0
    bankroll_path = []
    bets: List[Bet] = []

    n = len(df_test)
    for i in range(n):
        candidates = []

        # 1x2: avaliar H,D,A
        for k, sel in enumerate(["H", "D", "A"]):
            edge = float(p_model_1x2[i, k] - p_market_1x2[i, k])
            if edge >= min_edge:
                candidates.append(("1x2", sel, p_model_1x2[i, k], p_market_1x2[i, k], odds_1x2[i, k], edge))

        # OU: Over/Under
        edge_over = float(p_model_over[i] - p_market_over[i])
        if edge_over >= min_edge:
            candidates.append(("OU", "Over", float(p_model_over[i]), float(p_market_over[i]), float(odds_over[i]), edge_over))
        edge_under = float((1.0 - p_model_over[i]) - (1.0 - p_market_over[i]))
        if edge_under >= min_edge:
            candidates.append(("OU", "Under", float(1.0 - p_model_over[i]), float(1.0 - p_market_over[i]), float(odds_under[i]), edge_under))

        # ordenar por edge desc
        candidates.sort(key=lambda x: x[-1], reverse=True)
        candidates = candidates[:max_bets_per_game]

        # executar
        for (market, sel, pmod, pmkt, odds, edge) in candidates:
            if stake_mode == "flat":
                stake = flat_stake
            else:
                fk = kelly_fraction(pmod, odds)
                fk = min(fk, max_kelly)
                stake = bankroll * fk * fkelly

            # resultado
            if market == "1x2":
                won = 1 if (y_1x2[i] == {"H":0,"D":1,"A":2}[sel]) else 0
            else:
                won = 1 if ((y_over01[i] == 1 and sel == "Over") or (y_over01[i] == 0 and sel == "Under")) else 0

            pnl = stake * (odds - 1.0) if won else -stake
            bankroll += pnl

            bets.append(Bet(
                idx=int(df_test.index[i]),
                market=market,
                selection=sel,
                prob_model=float(pmod),
                prob_market=float(pmkt),
                odds=float(odds),
                edge=float(edge),
                stake=float(stake),
                won=int(won),
                pnl=float(pnl)
            ))

        bankroll_path.append({"row": int(df_test.index[i]), "bankroll": float(bankroll)})

    bets_df = pd.DataFrame([b.__dict__ for b in bets]) if bets else pd.DataFrame(columns=[f.name for f in Bet.__dataclass_fields__.values()])
    bankroll_df = pd.DataFrame(bankroll_path)
    return bets_df, bankroll_df

# ----------------------------
# Leitura de dataset (URL / local)
# ----------------------------
def read_dataset(path_or_url: str) -> pd.DataFrame:
    if path_or_url.lower().startswith("http"):
        # leitura via requests (sem depender de gdown)
        import requests
        r = requests.get(path_or_url)
        r.raise_for_status()
        content = r.content
        # inferir extensão
        if path_or_url.lower().endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        elif path_or_url.lower().endswith(".xlsx") or path_or_url.lower().endswith(".xls"):
            return pd.read_excel(io.BytesIO(content))
        else:
            # tentar csv por padrão
            try:
                return pd.read_csv(io.BytesIO(content))
            except Exception:
                return pd.read_excel(io.BytesIO(content))
    else:
        if path_or_url.lower().endswith(".csv"):
            return pd.read_csv(path_or_url)
        return pd.read_excel(path_or_url)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # reduzir espaços, padronizar
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", str(c).strip()) for c in df.columns]
    return df

def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Detecta colunas comuns. Você pode sobrescrever via args.
    Esperado:
      odds 1x2: (H,D,A)  -> ex: Odd_H, Odd_D, Odd_A, Home_Odds, Draw_Odds, Away_Odds
      odds OU: Over, Under -> ex: Odd_O, Odd_U, Over_Odds, Under_Odds
      gols: FTHG, FTAG (ou HomeGoals, AwayGoals)
      data: Date
    """
    cols = set(df.columns)
    def pick(candidates):
        for c in candidates:
            if c in cols: return c
        return ""

    mapping = {
        "odds_h": pick(["Odd_H","Odds_H","Home_Odds","H_Odds","oddH","H","ODDH","HomeOdd"]),
        "odds_d": pick(["Odd_D","Odds_D","Draw_Odds","D_Odds","oddD","D","ODDD","DrawOdd"]),
        "odds_a": pick(["Odd_A","Odds_A","Away_Odds","A_Odds","oddA","A","ODDA","AwayOdd"]),
        "odds_over": pick(["Odd_O","Odds_O","Over_Odds","O_Odds","oddO","Over","ODDO","OverOdd"]),
        "odds_under": pick(["Odd_U","Odds_U","Under_Odds","U_Odds","oddU","Under","ODDU","UnderOdd"]),
        "fthg": pick(["FTHG","HomeGoals","HG","FT_HG","Gols_Casa"]),
        "ftag": pick(["FTAG","AwayGoals","AG","FT_AG","Gols_Fora"]),
        "date": pick(["Date","DATA","MatchDate","Dia","data","date"]),
        # probabilidades já prontas (se existirem)
        "pH": pick(["pHome","pH","Prob_H","P_H"]),
        "pD": pick(["pDraw","pD","Prob_D","P_D"]),
        "pA": pick(["pAway","pA","Prob_A","P_A"]),
        "pOver": pick(["pOver","pO","Prob_Over","P_Over"]),
    }
    return mapping

# ----------------------------
# Modelagem
# ----------------------------
def fit_multinomial_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=2000,
        C=1.0,
        n_jobs=None
    )
    clf.fit(X, y)
    return clf

def fit_bin_logit(X: np.ndarray, y01: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=1.0
    )
    clf.fit(X, y01)
    return clf

def optimize_alpha_multiclass(p_mkt: np.ndarray, p_cal: np.ndarray, y: np.ndarray) -> float:
    alphas = np.linspace(0.0, 1.0, 51)
    best_a, best = 1.0, 1e18
    for a in alphas:
        p = a*p_mkt + (1-a)*p_cal
        ll = log_loss(y, p, labels=[0,1,2])
        if ll < best:
            best = ll
            best_a = float(a)
    return best_a

def optimize_alpha_binary(p_mkt: np.ndarray, p_cal: np.ndarray, y01: np.ndarray) -> float:
    alphas = np.linspace(0.0, 1.0, 51)
    best_a, best = 1.0, 1e18
    for a in alphas:
        p = a*p_mkt + (1-a)*p_cal
        ll = log_loss(y01, np.vstack([1-p, p]).T, labels=[0,1])
        if ll < best:
            best = ll
            best_a = float(a)
    return best_a

def knn_residual_adjustment(X_train: np.ndarray, resid_train: np.ndarray,
                            X_test: np.ndarray, k: int = 200, sigma: float = 0.08) -> np.ndarray:
    """
    Ajuste residual via KNN com kernel gaussiano no espaço de features (probabilidades derivadas).
    resid_train: y - p_base (para 1 classe / binário)
    Retorna delta para adicionar em p_base_test
    """
    k = min(k, len(X_train))
    if k <= 5:
        return np.zeros(len(X_test))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_train)
    dist, idx = nn.kneighbors(X_test, return_distance=True)
    w = np.exp(-(dist**2) / (2*(sigma**2)))
    wsum = w.sum(axis=1, keepdims=True) + 1e-12
    w = w / wsum
    delta = (w * resid_train[idx]).sum(axis=1)
    return delta

# ----------------------------
# Walk-forward
# ----------------------------
def walk_forward_splits(df: pd.DataFrame, min_train: int, step: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    splits = []
    start = min_train
    while start < n:
        train_idx = np.arange(0, start)
        test_end = min(n, start + step)
        test_idx = np.arange(start, test_end)
        splits.append((train_idx, test_idx))
        start = test_end
    return splits

# ----------------------------
# Plot helpers
# ----------------------------
def save_calibration_plot(calib_df: pd.DataFrame, title: str, outpath: str):
    x = calib_df["p_mean"].values
    y = calib_df["y_rate"].values
    m = ~np.isnan(x) & ~np.isnan(y)
    plt.figure()
    plt.plot([0,1],[0,1])
    plt.plot(x[m], y[m], marker="o")
    plt.xlabel("Probabilidade prevista (média do bin)")
    plt.ylabel("Frequência observada (média do bin)")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

def save_bankroll_plot(bankroll_df: pd.DataFrame, title: str, outpath: str):
    if bankroll_df.empty:
        return
    plt.figure()
    plt.plot(bankroll_df["bankroll"].values)
    plt.xlabel("Jogos (ordem temporal)")
    plt.ylabel("Bankroll")
    plt.title(title)
    plt.grid(True, alpha=0.2)
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-url", required=True, help="URL RAW do GitHub (csv/xlsx) ou caminho local.")
    ap.add_argument("--sheet", default=None, help="Se xlsx, nome da aba (opcional).")
    ap.add_argument("--date-col", default=None, help="Coluna de data (recomendado).")
    ap.add_argument("--ou-line", type=float, default=2.5, help="Linha do O/U (ex: 2.5).")
    ap.add_argument("--min-train", type=int, default=12000, help="Tamanho mínimo de treino para o 1º fold.")
    ap.add_argument("--step", type=int, default=2500, help="Tamanho do bloco de teste por fold.")
    ap.add_argument("--use-knn", action="store_true", help="Ativa ajuste residual KNN.")
    ap.add_argument("--knn-k", type=int, default=200)
    ap.add_argument("--knn-sigma", type=float, default=0.08)
    ap.add_argument("--min-edge", type=float, default=0.02, help="Edge mínimo vs mercado para apostar.")
    ap.add_argument("--stake-mode", choices=["flat","fkelly"], default="fkelly")
    ap.add_argument("--flat-stake", type=float, default=1.0)
    ap.add_argument("--fkelly", type=float, default=0.25, help="Multiplicador do Kelly (ex: 0.25 = 1/4 Kelly).")
    ap.add_argument("--max-kelly", type=float, default=0.03, help="Cap da fração Kelly por aposta.")
    ap.add_argument("--bankroll0", type=float, default=100.0)
    ap.add_argument("--max-bets-per-game", type=int, default=1)
    ap.add_argument("--outdir", default="out", help="Pasta de saída.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # carregar
    df = read_dataset(args.data_url)
    if args.sheet and args.data_url.lower().endswith((".xlsx",".xls")):
        df = pd.read_excel(args.data_url, sheet_name=args.sheet)
    df = normalize_columns(df)

    mapping = detect_columns(df)

    # data
    date_col = args.date_col or mapping.get("date") or ""
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # colunas essenciais
    fthg = mapping.get("fthg")
    ftag = mapping.get("ftag")
    if not fthg or not ftag or fthg not in df.columns or ftag not in df.columns:
        raise ValueError("Não encontrei colunas de gols (FTHG/FTAG). Ajuste o dataset para conter gols finais.")

    # odds / probs
    # 1x2
    has_probs_1x2 = all([mapping.get("pH") in df.columns if mapping.get("pH") else False,
                         mapping.get("pD") in df.columns if mapping.get("pD") else False,
                         mapping.get("pA") in df.columns if mapping.get("pA") else False])
    if has_probs_1x2:
        p1x2_mkt = df[[mapping["pH"], mapping["pD"], mapping["pA"]]].astype(float).values
        # garantir normalização
        p1x2_mkt = remove_margin_proportional(p1x2_mkt)
        odds_h = odds_d = odds_a = None
        odds_1x2 = None
    else:
        odds_h, odds_d, odds_a = mapping.get("odds_h"), mapping.get("odds_d"), mapping.get("odds_a")
        if not odds_h or not odds_d or not odds_a:
            raise ValueError("Não encontrei odds 1x2 (H/D/A) nem probabilidades pHome/pDraw/pAway.")
        odds_1x2 = df[[odds_h, odds_d, odds_a]].astype(float).values
        p1x2_mkt = odds_to_fair_probs_1x2(odds_1x2[:,0], odds_1x2[:,1], odds_1x2[:,2])

    # OU
    has_prob_over = mapping.get("pOver") and mapping["pOver"] in df.columns
    if has_prob_over:
        pover_mkt = df[mapping["pOver"]].astype(float).values
        pover_mkt = np.clip(pover_mkt, 1e-6, 1-1e-6)
        odds_over = odds_under = None
    else:
        odds_over_col, odds_under_col = mapping.get("odds_over"), mapping.get("odds_under")
        if not odds_over_col or not odds_under_col:
            raise ValueError("Não encontrei odds O/U (Over/Under) nem prob pOver.")
        odds_over = df[odds_over_col].astype(float).values
        odds_under = df[odds_under_col].astype(float).values
        pover_mkt = odds_to_fair_prob_over(odds_over, odds_under)

    # outcomes
    goals_home = df[fthg].astype(float).values
    goals_away = df[ftag].astype(float).values
    y_1x2 = np.where(goals_home > goals_away, 0, np.where(goals_home == goals_away, 1, 2)).astype(int)
    y_over01 = ((goals_home + goals_away) > args.ou_line).astype(int)

    # features
    X_df = market_features(p1x2_mkt, pover_mkt)
    X = X_df.values.astype(float)

    # odds arrays for simulation
    if odds_1x2 is None:
        # se não temos odds, não dá pra simular payoff — aborta simulação de 1x2
        odds_1x2 = np.full((len(df),3), np.nan)
    if odds_over is None or odds_under is None:
        odds_over = np.full(len(df), np.nan)
        odds_under = np.full(len(df), np.nan)

    # splits
    splits = walk_forward_splits(df, min_train=args.min_train, step=args.step)
    if len(splits) == 0:
        raise ValueError("Dataset pequeno demais para min_train/step. Ajuste parâmetros.")

    all_rows = []
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        # dentro do treino, separar validação temporal (últimos 20%)
        ntr = len(tr_idx)
        val_cut = int(ntr * 0.8)
        tr2_idx = tr_idx[:val_cut]
        val_idx = tr_idx[val_cut:]

        X_tr, X_val, X_te = X[tr2_idx], X[val_idx], X[te_idx]
        y_tr_1x2, y_val_1x2, y_te_1x2 = y_1x2[tr2_idx], y_1x2[val_idx], y_1x2[te_idx]
        y_tr_over, y_val_over, y_te_over = y_over01[tr2_idx], y_over01[val_idx], y_over01[te_idx]

        p_mkt_tr, p_mkt_val, p_mkt_te = p1x2_mkt[tr2_idx], p1x2_mkt[val_idx], p1x2_mkt[te_idx]
        po_mkt_tr, po_mkt_val, po_mkt_te = pover_mkt[tr2_idx], pover_mkt[val_idx], pover_mkt[te_idx]

        # calibradores
        m1 = fit_multinomial_logit(X_tr, y_tr_1x2)
        p_cal_val_1x2 = m1.predict_proba(X_val)
        a1 = optimize_alpha_multiclass(p_mkt_val, p_cal_val_1x2, y_val_1x2)

        p_cal_te_1x2 = m1.predict_proba(X_te)
        p_mix_te_1x2 = a1*p_mkt_te + (1-a1)*p_cal_te_1x2

        # OU
        m2 = fit_bin_logit(X_tr, y_tr_over)
        p_cal_val_over = m2.predict_proba(X_val)[:,1]
        a2 = optimize_alpha_binary(po_mkt_val, p_cal_val_over, y_val_over)

        p_cal_te_over = m2.predict_proba(X_te)[:,1]
        p_mix_te_over = a2*po_mkt_te + (1-a2)*p_cal_te_over

        # KNN residual adjustment (opcional)
        if args.use_knn:
            # usar X_tr2 completo (train+val?) para residual: melhor usar tr_idx inteiro para estabilidade
            X_knn = X[tr_idx]
            # resid para Over
            # base = mix (com alpha) no treino inteiro
            p_cal_knn_1x2 = m1.predict_proba(X_knn)
            p_mix_knn_1x2 = a1*p1x2_mkt[tr_idx] + (1-a1)*p_cal_knn_1x2
            # ajustar cada classe separadamente via resid
            adj_te = np.zeros_like(p_mix_te_1x2)
            for c in range(3):
                resid = ((y_1x2[tr_idx] == c).astype(float) - p_mix_knn_1x2[:,c])
                delta = knn_residual_adjustment(X_knn, resid, X_te, k=args.knn_k, sigma=args.knn_sigma)
                adj_te[:,c] = delta
            p_mix_te_1x2 = np.clip(p_mix_te_1x2 + adj_te, 1e-6, 1.0)
            p_mix_te_1x2 = _safe_div(p_mix_te_1x2, p_mix_te_1x2.sum(axis=1, keepdims=True))

            # Over
            p_cal_knn_over = m2.predict_proba(X_knn)[:,1]
            p_mix_knn_over = a2*pover_mkt[tr_idx] + (1-a2)*p_cal_knn_over
            resid_over = y_over01[tr_idx].astype(float) - p_mix_knn_over
            delta_over = knn_residual_adjustment(X_knn, resid_over, X_te, k=args.knn_k, sigma=args.knn_sigma)
            p_mix_te_over = np.clip(p_mix_te_over + delta_over, 1e-6, 1-1e-6)

        # métricas (teste)
        ll_mkt_1x2 = log_loss(y_te_1x2, p_mkt_te, labels=[0,1,2])
        ll_mod_1x2 = log_loss(y_te_1x2, p_mix_te_1x2, labels=[0,1,2])
        br_mkt_1x2 = brier_multiclass(y_te_1x2, p_mkt_te)
        br_mod_1x2 = brier_multiclass(y_te_1x2, p_mix_te_1x2)

        ll_mkt_ou = log_loss(y_te_over, np.vstack([1-po_mkt_te, po_mkt_te]).T, labels=[0,1])
        ll_mod_ou = log_loss(y_te_over, np.vstack([1-p_mix_te_over, p_mix_te_over]).T, labels=[0,1])
        br_mkt_ou = brier_binary(y_te_over, po_mkt_te)
        br_mod_ou = brier_binary(y_te_over, p_mix_te_over)

        fold_metrics.append({
            "fold": fold,
            "train_end_row": int(tr_idx[-1]),
            "test_start_row": int(te_idx[0]),
            "test_end_row": int(te_idx[-1]),
            "alpha_1x2": a1,
            "alpha_ou": a2,
            "logloss_mkt_1x2": ll_mkt_1x2,
            "logloss_mod_1x2": ll_mod_1x2,
            "brier_mkt_1x2": br_mkt_1x2,
            "brier_mod_1x2": br_mod_1x2,
            "logloss_mkt_ou": ll_mkt_ou,
            "logloss_mod_ou": ll_mod_ou,
            "brier_mkt_ou": br_mkt_ou,
            "brier_mod_ou": br_mod_ou,
        })

        # armazenar linhas
        for j, row_i in enumerate(te_idx):
            all_rows.append({
                "row": int(row_i),
                "fold": fold,
                "pH_mkt": float(p1x2_mkt[row_i,0]),
                "pD_mkt": float(p1x2_mkt[row_i,1]),
                "pA_mkt": float(p1x2_mkt[row_i,2]),
                "pH_mod": float(p_mix_te_1x2[j,0]),
                "pD_mod": float(p_mix_te_1x2[j,1]),
                "pA_mod": float(p_mix_te_1x2[j,2]),
                "pOver_mkt": float(pover_mkt[row_i]),
                "pOver_mod": float(p_mix_te_over[j]),
                "y1x2": int(y_1x2[row_i]),
                "yOver": int(y_over01[row_i]),
            })

    pred_df = pd.DataFrame(all_rows).sort_values("row").reset_index(drop=True)
    metrics_df = pd.DataFrame(fold_metrics)

    # salvar previsões e métricas
    pred_path = os.path.join(args.outdir, "predictions_walkforward.csv")
    met_path = os.path.join(args.outdir, "fold_metrics.csv")
    pred_df.to_csv(pred_path, index=False)
    metrics_df.to_csv(met_path, index=False)

    # calibração global (O/U) no conjunto test agregado
    calib_mkt = calibration_bins(pred_df["yOver"].values, pred_df["pOver_mkt"].values, n_bins=10)
    calib_mod = calibration_bins(pred_df["yOver"].values, pred_df["pOver_mod"].values, n_bins=10)
    calib_mkt.to_csv(os.path.join(args.outdir, "calibration_bins_ou_market.csv"), index=False)
    calib_mod.to_csv(os.path.join(args.outdir, "calibration_bins_ou_model.csv"), index=False)

    save_calibration_plot(calib_mkt, "Calibração O/U (Mercado - Closing Fair)", os.path.join(args.outdir, "calibration_ou_market.png"))
    save_calibration_plot(calib_mod, "Calibração O/U (Modelo Híbrido)", os.path.join(args.outdir, "calibration_ou_model.png"))

    # simulação de apostas usando as linhas de teste (precisa odds disponíveis)
    # Reconstituir arrays no mesmo "row" do pred_df
    test_rows = pred_df["row"].values.astype(int)
    df_test = df.iloc[test_rows].copy()

    p_mkt_1x2 = pred_df[["pH_mkt","pD_mkt","pA_mkt"]].values
    p_mod_1x2 = pred_df[["pH_mod","pD_mod","pA_mod"]].values
    p_mkt_over = pred_df["pOver_mkt"].values
    p_mod_over = pred_df["pOver_mod"].values
    y1x2 = pred_df["y1x2"].values.astype(int)
    yover = pred_df["yOver"].values.astype(int)

    # odds
    # tentar extrair novamente das colunas detectadas
    odds_1x2_test = np.full((len(df_test),3), np.nan)
    if mapping.get("odds_h") in df.columns and mapping.get("odds_d") in df.columns and mapping.get("odds_a") in df.columns:
        odds_1x2_test = df_test[[mapping["odds_h"], mapping["odds_d"], mapping["odds_a"]]].astype(float).values

    odds_over_test = np.full(len(df_test), np.nan)
    odds_under_test = np.full(len(df_test), np.nan)
    if mapping.get("odds_over") in df.columns and mapping.get("odds_under") in df.columns:
        odds_over_test = df_test[mapping["odds_over"]].astype(float).values
        odds_under_test = df_test[mapping["odds_under"]].astype(float).values

    bets_df, bankroll_df = simulate_bets(
        df_test=df_test,
        p_market_1x2=p_mkt_1x2,
        p_model_1x2=p_mod_1x2,
        odds_1x2=odds_1x2_test,
        y_1x2=y1x2,
        p_market_over=p_mkt_over,
        p_model_over=p_mod_over,
        odds_over=odds_over_test,
        odds_under=odds_under_test,
        y_over01=yover,
        min_edge=args.min_edge,
        stake_mode=args.stake_mode,
        flat_stake=args.flat_stake,
        fkelly=args.fkelly,
        max_kelly=args.max_kelly,
        max_bets_per_game=args.max_bets_per_game,
        bankroll0=args.bankroll0
    )
    bets_df.to_csv(os.path.join(args.outdir, "bets_simulated.csv"), index=False)
    bankroll_df.to_csv(os.path.join(args.outdir, "bankroll_path.csv"), index=False)
    save_bankroll_plot(bankroll_df, f"Bankroll (min_edge={args.min_edge}, stake={args.stake_mode})", os.path.join(args.outdir, "bankroll.png"))

    # resumo final
    def agg_metrics(prefix):
        return {
            f"{prefix}_mean": float(metrics_df[prefix].mean()),
            f"{prefix}_std": float(metrics_df[prefix].std(ddof=1)) if len(metrics_df)>1 else 0.0
        }

    summary = {
        "rows_total": int(len(df)),
        "rows_test_agg": int(len(pred_df)),
        "folds": int(len(metrics_df)),
        **agg_metrics("logloss_mkt_1x2"),
        **agg_metrics("logloss_mod_1x2"),
        **agg_metrics("brier_mkt_1x2"),
        **agg_metrics("brier_mod_1x2"),
        **agg_metrics("logloss_mkt_ou"),
        **agg_metrics("logloss_mod_ou"),
        **agg_metrics("brier_mkt_ou"),
        **agg_metrics("brier_mod_ou"),
        "alpha_1x2_mean": float(metrics_df["alpha_1x2"].mean()),
        "alpha_ou_mean": float(metrics_df["alpha_ou"].mean()),
        "bets_count": int(len(bets_df)),
        "bets_roi": float(bets_df["pnl"].sum() / (bets_df["stake"].sum() + 1e-12)) if len(bets_df) else 0.0,
        "final_bankroll": float(bankroll_df["bankroll"].iloc[-1]) if len(bankroll_df) else float(args.bankroll0),
    }
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # markdown report
    md = []
    md.append("# Relatório — Hybrid Closing Line (1x2 + O/U)\n")
    if date_col:
        md.append(f"- Ordenação temporal: `{date_col}`\n")
    else:
        md.append("- Ordenação temporal: **(não informada)** — recomendado incluir coluna de data.\n")
    md.append(f"- Folds: {summary['folds']}\n")
    md.append(f"- Teste agregado: {summary['rows_test_agg']} linhas\n")
    md.append("\n## Métricas (média ± desvio)\n")
    md.append(f"- LogLoss 1x2 — Mercado: {summary['logloss_mkt_1x2_mean']:.6f} ± {summary['logloss_mkt_1x2_std']:.6f}\n")
    md.append(f"- LogLoss 1x2 — Modelo : {summary['logloss_mod_1x2_mean']:.6f} ± {summary['logloss_mod_1x2_std']:.6f}\n")
    md.append(f"- Brier 1x2   — Mercado: {summary['brier_mkt_1x2_mean']:.6f} ± {summary['brier_mkt_1x2_std']:.6f}\n")
    md.append(f"- Brier 1x2   — Modelo : {summary['brier_mod_1x2_mean']:.6f} ± {summary['brier_mod_1x2_std']:.6f}\n")
    md.append(f"- LogLoss O/U — Mercado: {summary['logloss_mkt_ou_mean']:.6f} ± {summary['logloss_mkt_ou_std']:.6f}\n")
    md.append(f"- LogLoss O/U — Modelo : {summary['logloss_mod_ou_mean']:.6f} ± {summary['logloss_mod_ou_std']:.6f}\n")
    md.append(f"- Brier O/U   — Mercado: {summary['brier_mkt_ou_mean']:.6f} ± {summary['brier_mkt_ou_std']:.6f}\n")
    md.append(f"- Brier O/U   — Modelo : {summary['brier_mod_ou_mean']:.6f} ± {summary['brier_mod_ou_std']:.6f}\n")
    md.append("\n## Shrinkage\n")
    md.append(f"- alpha_1x2 (média): {summary['alpha_1x2_mean']:.3f}\n")
    md.append(f"- alpha_ou  (média): {summary['alpha_ou_mean']:.3f}\n")
    md.append("\n## Simulação de apostas (teórico no closing)\n")
    md.append(f"- min_edge: {args.min_edge}\n")
    md.append(f"- stake_mode: {args.stake_mode}\n")
    md.append(f"- bets: {summary['bets_count']}\n")
    md.append(f"- ROI: {summary['bets_roi']*100:.2f}%\n")
    md.append(f"- Bankroll final: {summary['final_bankroll']:.2f}\n")
    md.append("\n## Arquivos gerados\n")
    md.append("- predictions_walkforward.csv\n- fold_metrics.csv\n- calibration_bins_ou_market.csv\n- calibration_bins_ou_model.csv\n- calibration_ou_market.png\n- calibration_ou_model.png\n- bets_simulated.csv\n- bankroll_path.csv\n- bankroll.png\n- summary.json\n")
    with open(os.path.join(args.outdir, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("OK. Saída em:", args.outdir)
    print("Resumo:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
