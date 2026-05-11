import streamlit as st
import pandas as pd
import numpy as np

# Configuração da página
st.set_page_config(page_title="Validação Completa", layout="wide")

## Título e Estilo Educativo
st.title("📊 Validação Completa")
st.markdown("""
Este painel centraliza seus métodos validados. Utilize as abas para alternar entre 
diferentes estratégias e consultar as métricas de desempenho.
""")

---

# Sidebar para Navegação e Filtros
st.sidebar.header("Configurações do Método")
metodo_selecionado = st.sidebar.selectbox(
    "Selecione o Método",
    ["Under 2.5 Gols", "Match Odds", "Ambas Marcam", "True Odds (Calculadora)"]
)

---

## Área Principal: Dashboards de Validação

if metodo_selecionado == "True Odds (Calculadora)":
    st.subheader("🧮 Calculadora de Probabilidades Reais")
    col1, col2 = st.columns(2)
    
    with col1:
        # Inputs baseados no seu modelo Poisson/Dixon-Coles
        st.info("Insira os dados do confronto para calcular a True Odd.")
        home_expectancy = st.number_input("Expectativa de Gols (Casa)", min_value=0.0, value=1.20)
        away_expectancy = st.number_input("Expectativa de Gols (Fora)", min_value=0.0, value=0.90)
    
    with col2:
        # Lógica simplificada de exibição (conforme solicitado: objetiva e educativa)
        prob_vitoria = (home_expectancy / (home_expectancy + away_expectancy)) * 100
        true_odd = 100 / prob_vitoria if prob_vitoria > 0 else 0
        
        st.metric("Probabilidade Estimada", f"{prob_vitoria:.2f}%")
        st.success(f"True Odd Calculada: {true_odd:.2f}")

elif metodo_selecionado == "Under 2.5 Gols":
    st.subheader("📉 Método: Under 2.5 Gols")
    st.write("Histórico de validação e 'Drop Odds' para o mercado de menos de 3 gols.")
    
    # Exemplo de tabela de dados (Substituir pelo carregamento do arquivo .xlsm se necessário)
    data = {
        "Data": ["10/05/2026", "09/05/2026"],
        "Confronto": ["Time A x Time B", "Time C x Time D"],
        "Odd Entrada": [1.90, 1.85],
        "Resultado": ["Green", "Red"]
    }
    df = pd.DataFrame(data)
    st.table(df)

---

## Seção de Insights Educativos
with st.expander("💡 Conceitos de Validação"):
    st.write("""
    1. **Valor Esperado (+EV):** Uma aposta tem valor quando a odd oferecida é maior que a True Odd calculada.
    2. **Amostra:** Um método só é considerado validado após uma amostra mínima (ex: 500 entradas).
    3. **Dixon-Coles:** Ajuste técnico para corrigir a subestimação de empates em modelos Poisson simples.
    """)