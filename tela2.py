from futpythontrader import *

# Importando as Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from datetime import date
from rename import *
from leagues import *
from ligas_betfair import *

# Funções para carregar dados do GitHub privado

@st.cache_data
def load_data_betfair(dia):
    """Função para carregar o dataset Betfair do GitHub usando o token de autenticação."""
    file_path = f"https://github.com/futpythontrader/Jogos_do_Dia/raw/refs/heads/main/Betfair/Jogos_do_Dia_Betfair_Back_Lay_{dia}.csv"
    try:
        betfair = pd.read_csv(file_path)
        return betfair
    except:
        st.error("Jogos do Dia ainda não estão disponíveis.")
        return pd.DataFrame()


@st.cache_data
def load_data_base():
    """Função para carregar a base de dados FlashScore do GitHub usando o token de autenticação."""
    file_path = f"https://github.com/futpythontrader/Bases_de_Dados/raw/refs/heads/main/Betfair/Base_de_Dados_Betfair_Exchange_Back_Lay.csv"
    base_betfair = pd.read_csv(file_path)
    return base_betfair

# Funções de processamento de dados

def drop_reset_index(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df.index += 1
    return df

def ajustar_id_mercado(id_mercado, comprimento_decimal_desejado=9):
    id_mercado_str = str(id_mercado)
    partes = id_mercado_str.split('.')
    if len(partes) == 1:
        return id_mercado_str + '.' + '0' * comprimento_decimal_desejado
    parte_inteira, parte_decimal = partes
    zeros_para_adicionar = comprimento_decimal_desejado - len(parte_decimal)
    if zeros_para_adicionar > 0:
        parte_decimal += '0' * zeros_para_adicionar
    id_mercado_ajustado = parte_inteira + '.' + parte_decimal
    return id_mercado_ajustado

def remove_outliers(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.05)
        Q3 = df[col].quantile(0.95)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def entropy(probabilities):
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# Iniciando a Tela 2
def show_tela2():
    st.title("FutPythonTrader")
    st.header("Lay Away")

    dia = st.date_input("Data de Análise", date.today())

    # Carregar o dataset Betfair do dia selecionado
    df = load_data_betfair(dia)
    # df = df[df['League'].isin(ligas_betfair)]
    df = drop_reset_index(df)
    rename_leagues(df)
    df = df[df['League'].isin(leagues)]
    rename_teams(df)

    df['VAR1'] = np.sqrt((df['Odd_H_Back'] - df['Odd_A_Back'])**2)
    df['VAR2'] = np.degrees(np.arctan((df['Odd_A_Back'] - df['Odd_H_Back']) / 2))
    df['VAR3'] = np.degrees(np.arctan((df['Odd_D_Back'] - df['Odd_A_Back']) / 2))
    
    odds_columns = [col for col in df.columns if 'Odd_' in col]

    df_clean = remove_outliers(df, odds_columns)
    df = drop_reset_index(df_clean)

    cs_lay_columns = [col for col in df.columns if 'CS' in col and 'Lay' in col]
    cs_lay_data = df[cs_lay_columns]
    cv_cs_lay = cs_lay_data.apply(lambda x: x.std() / x.mean(), axis=1)
    df['CV_CS'] = cv_cs_lay

    probabilities_cs = cs_lay_data.replace(0, np.nan).apply(lambda x: 1 / x, axis=1)
    entropy_cs = probabilities_cs.apply(lambda x: -np.sum(x * np.log2(x)) if x.sum() != 0 else 0, axis=1)
    df['Entropy_CS'] = entropy_cs
    
    flt = (df.VAR1 >= 4) & (df.VAR2 >= 60) & (df.VAR3 <= -60)
    Entradas = df[flt]
    Entradas = drop_reset_index(Entradas)
    Entradas_Today = Entradas[['Date','Time','League','Home','Away','Odd_A_Lay']]
    
    st.subheader("Entradas")
    st.dataframe(Entradas_Today)

    st.subheader("")

    # # Carregar a base de dados Betfair para a data selecionada
    # base = load_data_base()
    # flt = base.Date == str(dia)
    # base_today = base[flt]
    # base_today = base_today[['League','Home','Away','Goals_H','Goals_A','Goals_Min_H','Goals_Min_A']]
    # base_today = drop_reset_index(base_today)
    
    # if not base_today.empty:
    #     Entradas_Resultado = pd.merge(Entradas, base_today, on=['League','Home', 'Away'])
    #     Entradas_Resultado = drop_reset_index(Entradas_Resultado)
    #     Entradas_Resultado['Profit'] = np.where((Entradas_Resultado['Goals_H'] >= Entradas_Resultado['Goals_A']), 
    #                                             9.4 / (Entradas_Resultado['Odd_A_Lay']-1), -10)
    #     Entradas_Resultado['Profit_Acu'] = Entradas_Resultado['Profit'].cumsum()
    #     Entradas_Resultado = Entradas_Resultado[['League','Home','Away','Goals_H','Goals_A','Goals_Min_H','Goals_Min_A','Profit','Profit_Acu']]
    #     st.subheader("Resultados das Entradas")
    #     st.dataframe(Entradas_Resultado)

    for a, b, c, d, e, f in zip(Entradas.League, Entradas.Time, Entradas.Home, 
                                Entradas.Away, Entradas.Odd_A_Lay, Entradas.IDMercado_Match_Odds):
        liga = a
        horario = b
        home = c
        away = d
        odd = e
        id_mercado = ajustar_id_mercado(f)

        st.write(f"Liga: {liga}")
        st.write(f"Jogo: {home} x {away}")
        st.write(f"Horário: {horario}")
        st.write(f"Odd: {odd}")
        link = f'<div style="text-align:left"><a href="https://www.betfair.com/exchange/plus/football/market/{id_mercado}">{"Betfair"}</a></div>'
        st.markdown(link, unsafe_allow_html=True)
        st.write('')
        st.write('')
        st.write('')
