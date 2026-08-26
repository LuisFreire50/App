import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
EPS = 1e-12


@dataclass
class ColumnMap:
    date: str
    target: str
    odd_h: str
    odd_d: str
    odd_a: str
    odd_over25: str
    odd_under25: str
    odd_btts_yes: str
    odd_btts_no: str
    odd_00_lay: str
    odd_01_lay: str
    odd_10_lay: str


def cv_pop(values):
    a = np.asarray(values, dtype=float)
    if len(a) == 0 or np.any(~np.isfinite(a)) or np.any(a <= 0):
        return np.nan
    m = a.mean()
    return np.nan if m <= 0 else a.std(ddof=0) / m


def remove_vig(odds):
    a = np.asarray(odds, dtype=float)
    if len(a) == 0 or np.any(~np.isfinite(a)) or np.any(a <= 1):
        return np.full(len(a), np.nan)
    inv = 1.0 / a
    s = inv.sum()
    return np.full(len(a), np.nan) if s <= 0 else inv / s


def entropy_normalized(probs):
    p = np.asarray(probs, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) <= 1:
        return np.nan
    return -np.sum(p * np.log(p)) / np.log(len(p))


def normalize_target(series):
    if pd.api.types.is_numeric_dtype(series):
        x = pd.to_numeric(series, errors="coerce")
        if set(x.dropna().unique()).issubset({0, 1}):
            return x.astype("Int64")
    text = series.astype(str).str.strip().str.lower()
    pos = {"1", "true", "yes", "green", "win", "winner", "won", "home", "success", "vitória", "vitoria", "ganhou"}
    neg = {"0", "false", "no", "red", "loss", "loser", "lost", "draw", "away", "fail", "derrota", "perdeu"}
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out[text.isin(pos)] = 1.0
    out[text.isin(neg)] = 0.0
    return out.astype("Int64")


def make_market_features(df, cm):
    out = df.copy()
    out["_date"] = pd.to_datetime(out[cm.date], errors="coerce", dayfirst=True)
    out["_target"] = normalize_target(out[cm.target])

    mapping = {
        "odd_h": cm.odd_h,
        "odd_d": cm.odd_d,
        "odd_a": cm.odd_a,
        "odd_over25": cm.odd_over25,
        "odd_under25": cm.odd_under25,
        "odd_btts_yes": cm.odd_btts_yes,
        "odd_btts_no": cm.odd_btts_no,
        "odd_00_lay": cm.odd_00_lay,
        "odd_01_lay": cm.odd_01_lay,
        "odd_10_lay": cm.odd_10_lay,
    }

    for new_col, src_col in mapping.items():
        out[new_col] = pd.to_numeric(out[src_col], errors="coerce")

    out["VAR50"] = out[["odd_h", "odd_d", "odd_a"]].apply(lambda r: cv_pop(r.values), axis=1)
    out["VAR51"] = out[["odd_over25", "odd_under25"]].apply(lambda r: cv_pop(r.values), axis=1)
    out["VAR52"] = out[["odd_btts_yes", "odd_btts_no"]].apply(lambda r: cv_pop(r.values), axis=1)
    out["VAR53"] = out[["odd_00_lay", "odd_01_lay", "odd_10_lay"]].apply(lambda r: cv_pop(r.values), axis=1)

    p1x2 = out.apply(lambda r: pd.Series(remove_vig([r["odd_h"], r["odd_d"], r["odd_a"]]),
                                         index=["p_home_novig", "p_draw_novig", "p_away_novig"]), axis=1)
    pou = out.apply(lambda r: pd.Series(remove_vig([r["odd_over25"], r["odd_under25"]]),
                                        index=["p_over25_novig", "p_under25_novig"]), axis=1)
    pbtts = out.apply(lambda r: pd.Series(remove_vig([r["odd_btts_yes"], r["odd_btts_no"]]),
                                          index=["p_btts_yes_novig", "p_btts_no_novig"]), axis=1)
    pscore = out.apply(lambda r: pd.Series(remove_vig([r["odd_00_lay"], r["odd_01_lay"], r["odd_10_lay"]]),
                                           index=["p00_cond", "p01_cond", "p10_cond"]), axis=1)

    out = pd.concat([out, p1x2, pou, pbtts, pscore], axis=1)

    out["VAR50_prob"] = out[["p_home_novig", "p_draw_novig", "p_away_novig"]].apply(lambda r: cv_pop(r.values), axis=1)
    out["entropy_1x2"] = out[["p_home_novig", "p_draw_novig", "p_away_novig"]].apply(lambda r: entropy_normalized(r.values), axis=1)

    out["home_away_gap"] = out["p_home_novig"] - out["p_away_novig"]
    out["home_draw_gap"] = out["p_home_novig"] - out["p_draw_novig"]
    out["home_away_logratio"] = np.log((out["p_home_novig"] + EPS) / (out["p_away_novig"] + EPS))

    out["over_bias"] = out["p_over25_novig"] - out["p_under25_novig"]
    out["VAR51_signed"] = (out["odd_under25"] - out["odd_over25"]) / (out["odd_under25"] + out["odd_over25"])

    out["btts_bias"] = out["p_btts_yes_novig"] - out["p_btts_no_novig"]
    out["VAR52_signed"] = (out["odd_btts_no"] - out["odd_btts_yes"]) / (out["odd_btts_no"] + out["odd_btts_yes"])

    out["score_home_bias"] = out["p10_cond"] - out["p01_cond"]
    out["score_home_away_logratio"] = np.log((out["p10_cond"] + EPS) / (out["p01_cond"] + EPS))
    out["score_00_concentration"] = out["p00_cond"]
    out["zero_vs_one_goal"] = out["p00_cond"] - (out["p01_cond"] + out["p10_cond"])

    out["interaction_home_var50"] = out["p_home_novig"] * out["VAR50"]
    out["interaction_home_over"] = out["home_away_gap"] * out["over_bias"]
    out["interaction_home_btts"] = out["home_away_gap"] * out["btts_bias"]
    out["interaction_home_score"] = out["home_away_gap"] * out["score_home_bias"]
    out["interaction_var50_entropy"] = out["VAR50"] * (1 - out["entropy_1x2"])

    return out


BASE_FEATURES = [
    "p_home_novig", "p_draw_novig", "p_away_novig",
    "p_over25_novig", "p_btts_yes_novig",
]

CV_FEATURES = BASE_FEATURES + ["VAR50", "VAR51", "VAR52", "VAR53"]

DIRECTION_FEATURES = CV_FEATURES + [
    "home_away_gap", "home_draw_gap", "over_bias", "btts_bias",
    "score_home_bias", "VAR51_signed", "VAR52_signed",
]

FULL_FEATURES = DIRECTION_FEATURES + [
    "VAR50_prob", "entropy_1x2", "home_away_logratio",
    "score_home_away_logratio", "score_00_concentration",
    "zero_vs_one_goal", "interaction_home_var50",
    "interaction_home_over", "interaction_home_btts",
    "interaction_home_score", "interaction_var50_entropy",
]


def expected_calibration_error(y_true, y_prob, bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (y_prob >= lo) & (y_prob <= hi) if i == len(edges) - 2 else (y_prob >= lo) & (y_prob < hi)
        if mask.sum():
            ece += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


def temporal_splits(n, n_splits=5):
    min_train = max(20, int(n * 0.50))
    remaining = n - min_train
    if remaining <= 0:
        return []
    fold_size = max(1, remaining // n_splits)
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * fold_size
        test_start = train_end
        test_end = n if i == n_splits - 1 else min(n, test_start + fold_size)
        if test_end > test_start:
            splits.append((np.arange(train_end), np.arange(test_start, test_end)))
    return splits


def make_models():
    models = {
        "LogisticRegression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=4000, C=1.0))
        ]),
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500, max_depth=8, min_samples_leaf=8,
                random_state=42, n_jobs=-1
            ))
        ]),
        "HistGradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=42
            ))
        ])
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1
            ))
        ])
    except Exception:
        pass

    return models


def evaluate_walk_forward(df, features, model, n_splits=5):
    data = df.sort_values("_date").reset_index().dropna(subset=["_date", "_target"]).copy()
    X = data[features]
    y = data["_target"].astype(int)

    preds = np.full(len(data), np.nan)
    fold_ids = np.full(len(data), -1)

    for fold, (train_idx, test_idx) in enumerate(temporal_splits(len(data), n_splits=n_splits)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]

        if y_train.nunique() < 2:
            continue

        estimator = clone(model)
        if len(train_idx) >= 100:
            estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)

        estimator.fit(X_train, y_train)
        preds[test_idx] = estimator.predict_proba(X_test)[:, 1]
        fold_ids[test_idx] = fold

    valid = np.isfinite(preds)
    yy = y.values[valid]
    pp = np.clip(preds[valid], 1e-6, 1 - 1e-6)

    auc = roc_auc_score(yy, pp) if len(np.unique(yy)) == 2 else np.nan

    pred_df = data.loc[valid, ["index", "_date", "_target", "odd_h", "p_home_novig"]].copy()
    pred_df["p_model"] = pp
    pred_df["fold"] = fold_ids[valid]

    metrics = {
        "Brier": brier_score_loss(yy, pp),
        "LogLoss": log_loss(yy, pp),
        "AUC": auc,
        "ECE": expected_calibration_error(yy, pp),
        "N_OOS": len(yy),
    }
    return metrics, pred_df


def kelly_full(odd, p):
    if not np.isfinite(odd) or not np.isfinite(p) or odd <= 1:
        return 0.0
    b = odd - 1
    q = 1 - p
    return max(0.0, (b * p - q) / b)


def kelly_multiplier(f):
    if f <= 0:
        return 0.0
    if f < 0.05:
        return 0.0
    if f < 0.08:
        return 0.50
    if f < 0.15:
        return 0.75
    if f <= 0.20:
        return 0.50
    return 0.25


def betting_backtest(pred_df, min_edge=0.05, max_stake=0.10):
    bt = pred_df.copy()
    bt["edge"] = bt["p_model"] - bt["p_home_novig"]
    bt["fair_odd"] = 1 / bt["p_model"]
    bt["kelly_full"] = [kelly_full(o, p) for o, p in zip(bt["odd_h"], bt["p_model"])]
    bt["kelly_multiplier"] = bt["kelly_full"].apply(kelly_multiplier)
    bt["stake_fraction"] = np.minimum(max_stake, bt["kelly_full"] * bt["kelly_multiplier"])
    bt["bet"] = (bt["edge"] >= min_edge) & (bt["stake_fraction"] > 0)
    bt.loc[~bt["bet"], "stake_fraction"] = 0.0
    bt["profit"] = np.where(
        bt["_target"].astype(int) == 1,
        bt["stake_fraction"] * (bt["odd_h"] - 1),
        -bt["stake_fraction"]
    )
    bt.loc[~bt["bet"], "profit"] = 0.0

    bets = bt[bt["bet"]].copy()
    volume = bets["stake_fraction"].sum()
    profit = bets["profit"].sum()
    roi = profit / volume if volume > 0 else np.nan

    equity = bets["profit"].cumsum()
    drawdown = equity - equity.cummax() if len(equity) else pd.Series(dtype=float)

    gross_win = bets.loc[bets["profit"] > 0, "profit"].sum()
    gross_loss = -bets.loc[bets["profit"] < 0, "profit"].sum()

    summary = {
        "Apostas": len(bets),
        "WinRate": (bets["_target"] == 1).mean() if len(bets) else np.nan,
        "Volume": volume,
        "Lucro": profit,
        "ROI": roi,
        "ProfitFactor": gross_win / gross_loss if gross_loss > 0 else np.nan,
        "MaxDrawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "EdgeMedio": bets["edge"].mean() if len(bets) else np.nan,
    }
    return bt, summary


def bootstrap_roi(bt, n_boot=2000, seed=42):
    bets = bt[bt["bet"]].copy()
    if len(bets) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = bets.iloc[rng.integers(0, len(bets), len(bets))]
        vol = sample["stake_fraction"].sum()
        if vol > 0:
            values.append(sample["profit"].sum() / vol)
    if not values:
        return np.nan, np.nan, np.nan
    a = np.asarray(values)
    return float(a.mean()), float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))


def fit_final_model(df, features, model):
    data = df.sort_values("_date").dropna(subset=["_date", "_target"]).copy()
    X = data[features]
    y = data["_target"].astype(int)
    estimator = clone(model)
    if len(data) >= 100:
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=3)
    estimator.fit(X, y)
    return estimator


def read_dataset(url):
    if "github.com" in url and "/blob/" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    return pd.read_csv(url)


def single_features(values):
    tmp = pd.DataFrame([values])
    tmp["date"] = pd.Timestamp.today()
    tmp["target"] = 0
    cm = ColumnMap(
        date="date", target="target",
        odd_h="odd_h", odd_d="odd_d", odd_a="odd_a",
        odd_over25="odd_over25", odd_under25="odd_under25",
        odd_btts_yes="odd_btts_yes", odd_btts_no="odd_btts_no",
        odd_00_lay="odd_00_lay", odd_01_lay="odd_01_lay", odd_10_lay="odd_10_lay"
    )
    return make_market_features(tmp, cm)


st.set_page_config(page_title="Validação Back Home", layout="wide")
st.title("Sistema de Validação Back Home")

url = st.text_input("URL do dataset CSV no GitHub")

if url:
    try:
        raw = read_dataset(url)
        st.success(f"Dataset carregado: {len(raw):,} linhas × {len(raw.columns):,} colunas")
        cols = list(raw.columns)

        with st.expander("Mapeamento das colunas", expanded=True):
            a, b, c = st.columns(3)
            with a:
                date_col = st.selectbox("Data", cols)
                target_col = st.selectbox("Target binário", cols)
                odd_h_col = st.selectbox("Odd Home Back", cols)
                odd_d_col = st.selectbox("Odd Draw Back", cols)
            with b:
                odd_a_col = st.selectbox("Odd Away Back", cols)
                odd_over_col = st.selectbox("Odd Over 2.5", cols)
                odd_under_col = st.selectbox("Odd Under 2.5", cols)
                odd_btts_yes_col = st.selectbox("Odd BTTS Yes", cols)
            with c:
                odd_btts_no_col = st.selectbox("Odd BTTS No", cols)
                odd_00_col = st.selectbox("Odd 0x0 Lay", cols)
                odd_01_col = st.selectbox("Odd 0x1 Lay", cols)
                odd_10_col = st.selectbox("Odd 1x0 Lay", cols)

        cm = ColumnMap(
            date=date_col, target=target_col,
            odd_h=odd_h_col, odd_d=odd_d_col, odd_a=odd_a_col,
            odd_over25=odd_over_col, odd_under25=odd_under_col,
            odd_btts_yes=odd_btts_yes_col, odd_btts_no=odd_btts_no_col,
            odd_00_lay=odd_00_col, odd_01_lay=odd_01_col, odd_10_lay=odd_10_col
        )

        engineered = make_market_features(raw, cm)

        x1, x2, x3 = st.columns(3)
        with x1:
            n_splits = st.slider("Folds temporais", 3, 10, 5)
        with x2:
            min_edge = st.slider("Edge mínimo", 0.00, 0.20, 0.05, 0.01)
        with x3:
            max_stake = st.slider("Stake máxima", 0.01, 0.10, 0.10, 0.01)

        models = make_models()
        selected_models = st.multiselect("Modelos", list(models.keys()), default=list(models.keys()))

        experiments = {
            "M0_BASE": BASE_FEATURES,
            "M1_CV": CV_FEATURES,
            "M2_DIRECTION": DIRECTION_FEATURES,
            "M3_FULL": FULL_FEATURES,
        }

        if st.button("Executar validação", type="primary"):
            metric_rows = []
            finance_rows = []

            for model_name in selected_models:
                for exp_name, features in experiments.items():
                    metrics, pred_df = evaluate_walk_forward(
                        engineered, features, models[model_name], n_splits=n_splits
                    )
                    bt, finance = betting_backtest(pred_df, min_edge=min_edge, max_stake=max_stake)
                    boot_mean, boot_lo, boot_hi = bootstrap_roi(bt)

                    metric_rows.append({"Modelo": model_name, "Experimento": exp_name, **metrics})
                    finance_rows.append({
                        "Modelo": model_name, "Experimento": exp_name, **finance,
                        "ROI_boot_mean": boot_mean,
                        "ROI_boot_2.5%": boot_lo,
                        "ROI_boot_97.5%": boot_hi,
                    })

            st.session_state["engineered"] = engineered
            st.session_state["models"] = models
            st.session_state["experiments"] = experiments
            st.session_state["metrics_df"] = pd.DataFrame(metric_rows)
            st.session_state["finance_df"] = pd.DataFrame(finance_rows)

        if "metrics_df" in st.session_state:
            metrics_df = st.session_state["metrics_df"]
            finance_df = st.session_state["finance_df"]

            st.subheader("Validação preditiva")
            st.dataframe(metrics_df.sort_values(["Brier", "LogLoss", "ECE"]), use_container_width=True)

            st.subheader("Validação financeira")
            st.dataframe(finance_df.sort_values("ROI", ascending=False), use_container_width=True)

            merged = metrics_df.merge(finance_df, on=["Modelo", "Experimento"])
            best = merged.sort_values(["Brier", "ECE", "ROI"], ascending=[True, True, False]).iloc[0]

            st.subheader("Modelo selecionado")
            st.dataframe(pd.DataFrame([best]), use_container_width=True)

            best_model_name = best["Modelo"]
            best_exp_name = best["Experimento"]

            st.subheader("Nova entrada")

            with st.form("new_market_form"):
                a, b, c = st.columns(3)
                with a:
                    odd_h = st.number_input("Odd Home Back", min_value=1.01, value=2.00, step=0.01)
                    odd_over25 = st.number_input("Odd Over 2.5", min_value=1.01, value=1.90, step=0.01)
                    odd_btts_yes = st.number_input("Odd BTTS Yes", min_value=1.01, value=1.85, step=0.01)
                    odd_00_lay = st.number_input("Odd 0x0 Lay", min_value=1.01, value=12.00, step=0.10)
                with b:
                    odd_d = st.number_input("Odd Draw Back", min_value=1.01, value=3.40, step=0.01)
                    odd_under25 = st.number_input("Odd Under 2.5", min_value=1.01, value=1.95, step=0.01)
                    odd_btts_no = st.number_input("Odd BTTS No", min_value=1.01, value=1.95, step=0.01)
                    odd_01_lay = st.number_input("Odd 0x1 Lay", min_value=1.01, value=11.00, step=0.10)
                with c:
                    odd_a = st.number_input("Odd Away Back", min_value=1.01, value=3.80, step=0.01)
                    odd_10_lay = st.number_input("Odd 1x0 Lay", min_value=1.01, value=8.00, step=0.10)

                submitted = st.form_submit_button("Calcular")

            if submitted:
                values = {
                    "odd_h": odd_h, "odd_d": odd_d, "odd_a": odd_a,
                    "odd_over25": odd_over25, "odd_under25": odd_under25,
                    "odd_btts_yes": odd_btts_yes, "odd_btts_no": odd_btts_no,
                    "odd_00_lay": odd_00_lay, "odd_01_lay": odd_01_lay, "odd_10_lay": odd_10_lay,
                }

                row = single_features(values)
                features = st.session_state["experiments"][best_exp_name]
                model = fit_final_model(
                    st.session_state["engineered"],
                    features,
                    st.session_state["models"][best_model_name]
                )

                p_model = float(model.predict_proba(row[features])[:, 1][0])
                p_market = float(row["p_home_novig"].iloc[0])
                edge = p_model - p_market
                fair_odd = 1 / p_model
                kf = kelly_full(odd_h, p_model)
                km = kelly_multiplier(kf)
                stake = min(max_stake, kf * km) if edge >= min_edge else 0.0

                result = pd.DataFrame([{
                    "Modelo": best_model_name,
                    "Experimento": best_exp_name,
                    "Prob_Modelo": p_model,
                    "Prob_Mercado_NoVig": p_market,
                    "Edge": edge,
                    "Odd_Justa": fair_odd,
                    "Odd_Mercado": odd_h,
                    "Kelly_Full": kf,
                    "Multiplicador_Kelly": km,
                    "Stake_Recomendada": stake,
                    "VAR50": float(row["VAR50"].iloc[0]),
                    "VAR51": float(row["VAR51"].iloc[0]),
                    "VAR52": float(row["VAR52"].iloc[0]),
                    "VAR53": float(row["VAR53"].iloc[0]),
                    "VAR50_prob": float(row["VAR50_prob"].iloc[0]),
                    "HomeAwayGap": float(row["home_away_gap"].iloc[0]),
                    "OverBias": float(row["over_bias"].iloc[0]),
                    "BTTSBias": float(row["btts_bias"].iloc[0]),
                    "ScoreHomeBias": float(row["score_home_bias"].iloc[0]),
                    "Entropy1X2": float(row["entropy_1x2"].iloc[0]),
                }])

                st.dataframe(result, use_container_width=True)

                if edge >= min_edge and stake > 0:
                    st.success("APOSTAR")
                else:
                    st.error("EVITAR")

    except Exception as e:
        st.exception(e)
