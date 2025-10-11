
import re, math, numpy as np, pandas as pd

def parse_c123(x):
    import numpy as _np
    import pandas as _pd
    if isinstance(x, (list, tuple, _np.ndarray)) and len(x) >= 3:
        c1, c2, c3 = x[0], x[1], x[2]
    elif isinstance(x, str):
        parts = re.split(r'[-,\|/;:\s]+', x.strip())
        parts = [p for p in parts if p != ""]
        if len(parts) >= 3:
            c1, c2, c3 = parts[0], parts[1], parts[2]
        else:
            c1 = c2 = c3 = 0
    elif _pd.isna(x):
        c1 = c2 = c3 = 0
    else:
        try:
            v = int(x)
            c1, c2, c3 = 0, 0, max(v, 0)
        except:
            c1 = c2 = c3 = 0
    try:
        return int(c1), int(c2), int(c3)
    except:
        return 0, 0, 0

def calc_R_quality(c1, c2, c3):
    num = c1 + 0.6*c2 + 0.3*c3
    den = (c1 + c2 + c3 + 1.0)
    return num / den

def calc_igt_plus(p1, p2, c1, c2, c3, ataques_perigosos_10):
    R = calc_R_quality(c1, c2, c3)
    p1_adj = p1 * (1 + 0.7*R)
    return 0.5*(p1_adj/50.0) + 0.3*(p2/50.0) + 0.2*min(ataques_perigosos_10/5.0, 1.2)

def goal_rate_lambda(igt_plus, base_lambda_per_min=0.028, k=2.2):
    return max(1e-4, base_lambda_per_min * math.exp(k*(igt_plus - 0.5)))

def prob_at_least_n_goals(time_minutes, lam_per_min, n=1):
    lamT = lam_per_min * max(time_minutes, 0)
    s = 0.0
    for k in range(n):
        s += math.exp(-lamT) * (lamT**k) / math.factorial(k)
    return max(0.0, min(1.0, 1.0 - s))

def fair_odds(p):
    return float('inf') if p <= 0 else round(1.0/max(p, 1e-6), 2)

def process_scanner_df(df):
    cols = {c.lower().strip(): c for c in df.columns}
    def get_col(name_alt):
        if isinstance(name_alt, (list, tuple)):
            for k in name_alt:
                key = k.lower().strip()
                if key in cols:
                    return cols[key]
            raise KeyError(f"Colunas não encontradas: {name_alt}")
        else:
            key = name_alt.lower().strip()
            if key in cols:
                return cols[key]
            raise KeyError(f"Coluna não encontrada: {name_alt}")

    c_min = get_col(['Minuto','minuto','min'])
    c_p1  = get_col(['Pressão1','pressão1','pressao1','p1'])
    c_p2  = get_col(['Pressão2','pressão2','pressao2','p2'])
    c_ap  = get_col(['Ataques Perigosos','ataques perigosos','ap','dangerous attacks'])
    c_c123= get_col(['Chutes no Gol C1C2C3','c1c2c3','c123'])
    c_sot = get_col(['Chutes no Gol','shots on target','sot'])
    c_ck  = get_col(['Escanteios','escanteios','corners','ck'])

    df_sorted = df.sort_values(by=c_min).reset_index(drop=True)
    df_sorted['AP_10m'] = df_sorted[c_ap].rolling(window=10, min_periods=1).sum()

    c1_list, c2_list, c3_list = [], [], []
    for v in df_sorted[c_c123].tolist():
        c1, c2, c3 = parse_c123(v)
        c1_list.append(c1); c2_list.append(c2); c3_list.append(c3)
    df_sorted['C1'] = c1_list
    df_sorted['C2'] = c2_list
    df_sorted['C3'] = c3_list

    igt_vals, lam_vals = [], []
    for p1, p2, ap10, c1, c2, c3 in zip(df_sorted[c_p1], df_sorted[c_p2], df_sorted['AP_10m'], df_sorted['C1'], df_sorted['C2'], df_sorted['C3']):
        igt = calc_igt_plus(float(p1), float(p2), int(c1), int(c2), int(c3), float(ap10))
        lam = goal_rate_lambda(igt)
        igt_vals.append(igt); lam_vals.append(lam)
    df_sorted['IGT_plus'] = igt_vals
    df_sorted['lambda_per_min'] = lam_vals

    def t_to_ht(minute): return max(0, 45 - int(minute))
    def t_to_ft(minute): return max(0, 90 - int(minute))

    p_10m, p_ht, p1_ft, p2_ft, p3_ft = [], [], [], [], []
    for m, lam in zip(df_sorted[c_min], df_sorted['lambda_per_min']):
        p_10m.append( prob_at_least_n_goals(10, lam, n=1) )
        p_ht.append(  prob_at_least_n_goals(t_to_ht(m), lam, n=1) )
        p1_ft.append( prob_at_least_n_goals(t_to_ft(m), lam, n=1) )
        p2_ft.append( prob_at_least_n_goals(t_to_ft(m), lam, n=2) )
        p3_ft.append( prob_at_least_n_goals(t_to_ft(m), lam, n=3) )

    df_sorted['P(gol_10m)']   = np.round(p_10m, 4)
    df_sorted['P(1+ até HT)'] = np.round(p_ht, 4)
    df_sorted['P(1+ até FT)'] = np.round(p1_ft, 4)
    df_sorted['P(2+ até FT)'] = np.round(p2_ft, 4)
    df_sorted['P(3+ até FT)'] = np.round(p3_ft, 4)

    df_sorted['Odd justa gol 10m']   = [round(1/max(p,1e-6),2) for p in df_sorted['P(gol_10m)']]
    df_sorted['Odd justa Over 0.5 HT'] = [round(1/max(p,1e-6),2) for p in df_sorted['P(1+ até HT)']]
    df_sorted['Odd justa 1+ FT']     = [round(1/max(p,1e-6),2) for p in df_sorted['P(1+ até FT)']]
    df_sorted['Odd justa 2+ FT']     = [round(1/max(p,1e-6),2) for p in df_sorted['P(2+ até FT)']]
    df_sorted['Odd justa 3+ FT']     = [round(1/max(p,1e-6),2) for p in df_sorted['P(3+ até FT)']]

    signals = []
    for igt, p10, pht, p2f in zip(df_sorted['IGT_plus'], df_sorted['P(gol_10m)'], df_sorted['P(1+ até HT)'], df_sorted['P(2+ até FT)']):
        sigs = []
        if igt >= 0.55 and pht >= 0.55:
            sigs.append('Over 0.5 HT')
        if igt >= 0.60 and p2f >= 0.60:
            sigs.append('Over 1.5 FT')
        if igt >= 0.70 and p10 >= 0.40:
            sigs.append('Over limite (próx 10m)')
        signals.append(", ".join(sigs) if sigs else "")
    df_sorted['Sinal'] = signals
    return df_sorted

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Caminho do CSV minuto a minuto')
    parser.add_argument('--out', type=str, default='igt_plus_output.csv')
    args = parser.parse_args()

    df_in = pd.read_csv(args.csv)
    df_out = process_scanner_df(df_in)
    df_out.to_csv(args.out, index=False)
    print('Arquivo salvo em:', args.out)
