import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
import math
from scipy.stats import binom, norm # Importar para distribuição binomial e normal

# --- Funções Auxiliares ---
def arange_inclusivo(inicio, fim, passo):
    return np.arange(inicio, fim + passo, passo)

def criterio_kelly(p, b):
    q = 1 - p
    return (b * p - q) / (b + 1e-9) # Adicionado 1e-9 para evitar divisão por zero

def verificar_vantagem(p, odd_fixa, distribuicoes):
    if odd_fixa is not None and odd_fixa > 0:
        b = odd_fixa - 1
        if b <= 0 or criterio_kelly(p, b) <= 0:
            return False
    elif distribuicoes:
        has_positive_kelly = False
        for dist in distribuicoes:
            if len(dist) == 4:
                _, inicio, fim, passo = dist
                odds = arange_inclusivo(inicio, fim, passo)
                valid_odds = odds[odds > 1]
                
                if len(valid_odds) > 0:
                    if np.any(criterio_kelly(p, valid_odds - 1) > 0):
                        has_positive_kelly = True
                        break
        if not has_positive_kelly:
            return False
    else:
        return False
    return True

def calcular_maior_sequencia_derrotas(taxa_acerto_decimal, tamanho_amostra):
    """
    Calcula a maior sequência esperada de derrotas (bad runs).
    """
    if not (0 < taxa_acerto_decimal < 1):
        return None

    rho = 1 / taxa_acerto_decimal

    if (rho - 1) <= 0:
        return None
    base_log = rho / (rho - 1)

    N = tamanho_amostra
    
    if N <= 0:
        return None
        
    try:
        resultado_log = math.log(N, base_log)
        maior_sequencia_esperada = round(resultado_log)
        return maior_sequencia_esperada
    except ValueError as e:
        return None

def calcular_esperanca_perder_k_consecutivas(n, k_input, p_acerto):
    """
    Calcula a quantidade de vezes esperada (e_k) de perder "k_input" apostas consecutivas
    em um espaço amostral de tamanho "n".
    Parâmetros:
    n (int): O tamanho da amostra (Número de Apostas).
    k_input (int): O número de apostas consecutivas a perder, escolhido pelo usuário.
    p_acerto (float): A probabilidade de acerto do evento (entre 0 e 1).
    Retorna:
    float: A quantidade esperada de vezes que "k_input" derrotas consecutivas ocorrem.
    Retorna None se os parâmetros forem inválidos.
    """
    if not (0 < p_acerto < 1) or n <= 0 or k_input <= 0:
        return None
    
    p_perda = 1 - p_acerto
    
    try:
        e_k = (n - k_input + 1) * (p_perda**k_input)
        return e_k
    except Exception:
        return None

def calculate_max_drawdowns(bankrolls_history):
    """
    Calcula o máximo drawdown para cada simulação.
    bankrolls_history: array numpy (num_simulacoes, num_apostas)
    Retorna: array numpy com os máximos drawdowns em porcentagem (0 a 100).
    """
    max_drawdowns = []
    for simulation_path in bankrolls_history:
        peak = simulation_path[0]
        max_dd_current_sim = 0.0
        for value in simulation_path:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd_current_sim:
                max_dd_current_sim = drawdown
        max_drawdowns.append(max_dd_current_sim * 100) # Convert to percentage
    return np.array(max_drawdowns)

# --- Nova Função para Mini Simulação Condicional ---
def mini_simulacao_condicional(
    bankroll_inicial,
    quantia_x,
    aposta_y,
    num_apostas_total,
    taxa_vitoria,
    odd_fixa,
    distribuicoes_odds,
    kelly_fracao,
    num_simulacoes_mini
):
    num_apostas_restantes = num_apostas_total - aposta_y

    if num_apostas_restantes <= 0:
        return None, None, None

    # --- Gerar todas as odds antecipadamente ---
    if odd_fixa is not None and odd_fixa > 1:
        odds = np.full((num_simulacoes_mini, num_apostas_restantes), odd_fixa)
    else:
        pesos_totais = sum(d[0] for d in distribuicoes_odds if len(d) == 4)
        if pesos_totais == 0:
            return None, None, None

        odds_lista = []
        for peso, inicio, fim, passo in distribuicoes_odds:
            if passo <= 0: passo = 0.01
            odds_segmento = arange_inclusivo(inicio, fim, passo)
            rep = int((peso / pesos_totais) * 100)
            odds_lista.extend(np.repeat(odds_segmento, rep))

        if len(odds_lista) == 0:
            return None, None, None

        odds_array = np.array(odds_lista)
        odds = np.random.choice(odds_array, size=(num_simulacoes_mini, num_apostas_restantes))

    # --- Gerar resultados (vitórias ou derrotas) ---
    resultados = np.random.rand(num_simulacoes_mini, num_apostas_restantes) < taxa_vitoria

    # --- Calcular Kelly e aposta em % ---
    b = odds - 1
    f_kelly = criterio_kelly(taxa_vitoria, b)
    f_aplicada = np.clip(f_kelly * kelly_fracao, 0, 0.999)

    # --- Inicializar bankrolls ---
    bankrolls = np.full(num_simulacoes_mini, quantia_x)

    # --- Simular apostas vetorizadas ---
    for i in range(num_apostas_restantes):
        aposta = bankrolls * f_aplicada[:, i]
        ganho = aposta * b[:, i]
        lucro = np.where(resultados[:, i], ganho, -aposta)
        bankrolls += lucro
        bankrolls = np.maximum(bankrolls, 0)

    # --- Resultados finais ---
    terminou_maior_igual_x = np.sum(bankrolls > quantia_x)
    chance_final_maior_igual_x = (terminou_maior_igual_x / num_simulacoes_mini) * 100

    terminou_maior_inicial = np.sum(bankrolls > bankroll_inicial)
    chance_final_maior_inicial = (terminou_maior_inicial / num_simulacoes_mini) * 100

    return chance_final_maior_igual_x, chance_final_maior_inicial, bankrolls

def plot_histograma_tricolor_mini_sim(bankrolls_finais_mini, quantia_x_mini_sim, bankroll_inicial_total):
    import plotly.graph_objects as go
    import numpy as np
    import streamlit as st

    # --- Calcular os dados do histograma manualmente ---
    hist, bin_edges = np.histogram(bankrolls_finais_mini, bins='auto')
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # --- Calcular percentuais para cada categoria ---
    total_simulacoes = len(bankrolls_finais_mini)
    
    # Contar simulações em cada categoria
    abaixo_partida = np.sum(bankrolls_finais_mini < quantia_x_mini_sim)
    entre_partida_inicial = np.sum((bankrolls_finais_mini >= quantia_x_mini_sim) & 
                                  (bankrolls_finais_mini < bankroll_inicial_total))
    acima_inicial_total = np.sum(bankrolls_finais_mini >= bankroll_inicial_total)
    
    # Calcular percentuais
    percent_abaixo = (abaixo_partida / total_simulacoes) * 100
    percent_entre = (entre_partida_inicial / total_simulacoes) * 100
    percent_acima = (acima_inicial_total / total_simulacoes) * 100

    # --- Colorir com base em três categorias ---
    colors = []
    for center in bin_centers:
        if center < quantia_x_mini_sim:
            colors.append('#E74C3C')  # Vermelho: abaixo do bankroll de partida
        elif center >= quantia_x_mini_sim and center < bankroll_inicial_total:
            colors.append('#F4D03F')  # Amarelo pálido: acima do bankroll de partida mas abaixo do inicial total
        else:
            colors.append('#27AE60')  # Verde: acima do bankroll inicial total

    # --- Criar figura com barras individuais coloridas ---
    fig = go.Figure()

    for i in range(len(hist)):
        fig.add_trace(go.Bar(
            x=[bin_centers[i]],
            y=[hist[i]],
            width=[bin_edges[i+1] - bin_edges[i]],
            marker_color=colors[i],
            marker_line_color="black",
            marker_line_width=1,
            showlegend=False
        ))

    # Adicionar legenda manual com percentuais
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='#E74C3C'),  # Tamanho reduzido
        legendgroup='red', 
        showlegend=True, 
        name=f'Prejuízo ({percent_abaixo:.1f}%)'  # Nome mais curto
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='#F4D03F'),  # Tamanho reduzido
        legendgroup='yellow', 
        showlegend=True, 
        name=f'Recuperação Parcial ({percent_entre:.1f}%)'  # Nome mais curto
    ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(size=12, color='#27AE60'),  # Tamanho reduzido
        legendgroup='green', 
        showlegend=True, 
        name=f'Recuperação Total ({percent_acima:.1f}%)'  # Nome mais curto
    ))

    fig.update_layout(
        title='Histograma Condicional dos Bankrolls Finais (Projeção Futura)',
        xaxis_title='Bankroll Final',
        yaxis_title='Contagem de Simulações',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=950,
        height=500,
        bargap=0.02,
        # Configurações para reduzir o espaço da legenda
        legend=dict(
            orientation="h",  # Legenda horizontal
            yanchor="top",
            y=1.1,  # Posiciona a legenda abaixo do gráfico
            xanchor="center",
            x=0.46,
            font=dict(size=12),  # Fonte menor
            bgcolor='rgba(255, 255, 255, 0.8)',  # Fundo semi-transparente
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1
        ),
        margin=dict(b=100)  # Aumenta a margem inferior para acomodar a legenda
    )

    st.plotly_chart(fig, use_container_width=True)




# URL da sua imagem GIF (substitua pela sua URL real)
gif_url = "https://i.imgur.com/Vl3k3gJ.gif" # Exemplo de GIF, substitua pela sua.

# --- Configuração da Página Streamlit ---
st.set_page_config(layout="wide", page_title="Simulação de Monte Carlo Otimizada")

# Aplicar cores de fundo usando CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8F8FF;
    }
    .stSidebar {
        background-color: #FFFFE0;
    }
    /* Estilo para as caixas do sumário */
    .summary-box {
        border: 1px solid #e0e0e0; /* Borda suave */
        border-radius: 8px; /* Cantos mais arredondados */
        padding: 20px; /* Mais padding */
        margin-bottom: 15px; /* Mais espaço entre as caixas */
        background-color: #ffffff; /* Fundo branco sólido */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra mais pronunciada */
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Espaço entre o label e o valor */
        align-items: flex-start;
        transition: transform 0.2s ease-in-out; /* Transição suave no hover */
    }
    .summary-box:hover {
        transform: translateY(-5px); /* Efeito de "levantar" no hover */
    }
    .summary-box .metric-label {
        font-weight: 600; /* Um pouco mais negrito */
        font-size: 1.05em; /* Tamanho levemente ajustado */
        margin-bottom: 8px; /* Mais espaço entre label e valor */
        color: #555555; /* Cor de texto mais suave */
    }
    .summary-box .metric-value {
        font-size: 2.2em; /* Valor maior para destaque */
        color: #333333; /* Cor mais escura para o valor */
        font-weight: 700; /* Mais negrito para o valor */
        margin-bottom: 5px; /* Espaço antes do delta */
    }
    .summary-box .metric-delta {
        font-size: 1em;
        display: flex;
        align-items: center;
        gap: 5px;
        font-weight: 500;
    }
    /* Cores do delta personalizado */
    .metric-delta-normal {
        color: #28a745; /* Verde Bootstrap */
    }
    .metric-delta-inverse {
        color: #dc3545; /* Vermelho Bootstrap */
    }
    .metric-delta-neutral {
        color: #6c757d; /* Cinza para neutro */
    }
    /* Ícones de seta aprimorados */
    .arrow-icon {
        font-size: 1.1em;
        line-height: 1;
    }
    /* Estilo para a projeção de crescimento */
    .growth-projection-box {
        font-size: 1.1em;
        text-align: center;
        margin-top: 30px; /* Mais espaço acima */
        padding: 15px; /* Mais padding */
        border: 1px solid #90EE90; /* Borda verde clara */
        border-radius: 8px;
        background-color: #e6ffe6; /* Fundo verde pastel */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        color: #333;
    }
    .growth-projection-box strong {
        color: #008000; /* Verde escuro para destaque */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Título com GIF e texto ---
st.markdown(f"<h1 style='display: flex; align-items: center; gap: 10px;'><img src='{gif_url}' width='45'> Simulação de Monte Carlo Otimizada</h1>", unsafe_allow_html=True)
st.markdown("Ajuste os parâmetros da simulação ao lado e clique em 'Executar Simulação' para ver os resultados.")

# --- Sidebar para Parâmetros Personalizáveis ---
st.sidebar.header("⚙️ Parâmetros da Simulação")

p_val = st.sidebar.slider("Probabilidade de Acerto (p)", 0.01, 0.99, 0.81, 0.01)
bankroll_inicial_val = st.sidebar.number_input("Bankroll Inicial", 10.0, 100000.0, 100.0, 10.0)
num_simulacoes_val = st.sidebar.number_input("Número de Simulações", 100, 100000, 10000, 100)
num_apostas_val = st.sidebar.number_input("Número de Apostas", 1, 1000, 30, 1)
fracao_kelly_val = st.sidebar.slider("Fração de Kelly", 0.01, 1.0, 0.33, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("🔴 Parâmetros de Sequência de Derrotas")
# Novo input para K
k_derrotas_input = st.sidebar.number_input(
    "Número de derrotas consecutivas (k)",
    min_value=1,
    max_value=num_apostas_val, # k não pode ser maior que o número total de apostas
    value=min(3, num_apostas_val), # Valor padrão 3, mas limitado ao num_apostas_val
    step=1,
    help="Define o número de derrotas consecutivas 'k' para a expectativa e cálculo de bad runs."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📉 Parâmetros de Drawdown")
drawdown_limite_val = st.sidebar.slider(
    "Probabilidade de Drawdown ≥ X% (Limite)",
    0.0, 99.0, 20.0, 1.0,
    help="Define o limite de drawdown para colorir o histograma e calcular a chance de ultrapassá-lo."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Projeção de Crescimento")
target_growth_percentage = st.sidebar.slider(
    "Alvo de Aumento de Saldo (%)",
    1, 100, 50, 1, # De 1% a 100%, passo de 1%
    help="Defina o percentual de aumento do saldo desejado para calcular a projeção de apostas."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Projeção Futura Condicional")
aposta_y_mini_sim = st.sidebar.number_input(
    "Aposta de Início da Projeção Futura (Y)",
    min_value=0,
    max_value=num_apostas_val,
    value=min(10, num_apostas_val),
    step=1,
    help="Em qual número de aposta a projeção futura deve começar?"
)
quantia_x_mini_sim = st.sidebar.number_input(
    "Bankroll de Início da Projeção Futura (X)",
    min_value=0.0,
    max_value=bankroll_inicial_val * 10, # Limite superior ajustado para ser mais flexível
    value=bankroll_inicial_val,
    step=bankroll_inicial_val * 0.1,
    format="%.2f",
    help="Com qual bankroll a projeção futura deve começar na Aposta Y?"
)
num_simulacoes_mini = st.sidebar.number_input(
    "Número de Simulações",
    min_value=100,
    max_value=100000,
    value=10000,
    step=1000,
    help="Quantas simulações devem ser executadas a partir do ponto (X, Y)?"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Configuração de Odds")

odd_choice = st.sidebar.radio("Escolha o tipo de Odd:", ("Odds Fixas", "Distribuição de Odds"))

odd_fixa_val = None
distribuicoes_val = []

if odd_choice == "Odds Fixas":
    odd_fixa_val = st.sidebar.number_input("Odd Fixa", 1.01, 10.0, 1.30, 0.01)
else:
    st.sidebar.markdown("Adicione distribuições de odds. Formato: `[peso %, odd_inicio, odd_fim, odd_passo]`")
    num_distribuicoes_input = st.sidebar.number_input("Número de Distribuições", 1, 5, 1, help="Defina o número de distribuições de odds para incluir na simulação.")
    
    default_dist_configs = [
        [10, 1.31, 1.33, 0.01],
        [85, 1.28, 1.30, 0.01],
        [5, 1.34, 1.36, 0.01],
        [50, 1.20, 1.22, 0.01],
        [50, 1.35, 1.37, 0.01]
    ]

    for i in range(num_distribuicoes_input):
        st.sidebar.subheader(f"Distribuição {i+1}")
        
        default_pct = default_dist_configs[i][0] if i < len(default_dist_configs) else 100 // num_distribuicoes_input
        default_inicio = default_dist_configs[i][1] if i < len(default_dist_configs) else 1.28
        default_fim = default_dist_configs[i][2] if i < len(default_dist_configs) else 1.30
        default_passo = default_dist_configs[i][3] if i < len(default_dist_configs) else 0.01

        col1, col2, col3, col4 = st.sidebar.columns(4)
        with col1:
            pct = st.number_input(f"Peso % {i+1}", 1, 100, default_pct, key=f"pct_{i}", help="Peso percentual para esta faixa de odds na simulação.")
        with col2:
            inicio = st.number_input(f"Início {i+1}", 1.01, 5.0, default_inicio, 0.01, key=f"inicio_{i}", help="Valor inicial da odd para esta distribuição.")
        with col3:
            fim = st.number_input(f"Fim {i+1}", 1.01, 5.0, default_fim, 0.01, key=f"fim_{i}", help="Valor final da odd para esta distribuição.")
        with col4:
            passo = st.number_input(f"Passo {i+1}", 0.001, 0.1, default_passo, 0.001, format="%.3f", key=f"passo_{i}", help="Intervalo entre as odds (ex: 0.01 para 1.28, 1.29, 1.30).")
        
        distribuicoes_val.append([pct, inicio, fim, passo])

st.sidebar.markdown("---")

# --- Botão de Execução (REMOVIDA A LINHA E ADICIONADO O ÍCONE) ---
if st.button("🎲 Executar Simulação"):
    st.markdown("---")
    st.header("📝 Sumário dos Resultados Financeiros")

    # --- Verificação de Vantagem ---
    try:
        if not verificar_vantagem(p_val, odd_fixa_val, distribuicoes_val if odd_choice == "Distribuição de Odds" else []):
            st.error("🚫 Impossível continuar! Vantagem esperada nula ou negativa. Ajuste seus parâmetros para garantir uma vantagem positiva.")
            st.stop()
    except Exception as e:
        st.error(f"⚠️ Erro na verificação de vantagem: {e}. Verifique os parâmetros de odds.")
        st.stop()

    # --- Cálculo da Maior Sequência Esperada de Derrotas ---
    maior_sequencia_derrotas = calcular_maior_sequencia_derrotas(p_val, num_apostas_val)

    # --- Cálculo da Quantidade Esperada de Perder K Consecutivas ---
    # Agora usamos o 'k_derrotas_input' diretamente
    esperanca_perder_k_consecutivas = calcular_esperanca_perder_k_consecutivas(num_apostas_val, k_derrotas_input, p_val)


    # --- Simulação de Monte Carlo Otimizada ---
    odds_possiveis = []
    if odd_fixa_val is not None:
        odds_possiveis = np.array([odd_fixa_val], dtype=np.float32)
    else:
        total_pct = sum(d[0] for d in distribuicoes_val)
        if total_pct == 0:  
            st.error("🚨 Todos os pesos percentuais das distribuições são zero. Adicione pesos válidos para continuar.")
            st.stop()

        for pct, i, f, s in distribuicoes_val:
            if s <= 0:
                s = 0.01  
                st.warning(f"⚠️ Aviso: Passo da distribuição inválido (0 ou negativo). Ajustando para 0.01.")
            odds_segment = arange_inclusivo(i, f, s)
            
            num_repetitions = int(pct / total_pct * 100)
            odds_possiveis.extend(np.repeat(odds_segment, num_repetitions))

        odds_possiveis = np.array(odds_possiveis, dtype=np.float32)
    
    if len(odds_possiveis) == 0:
        st.error("🚫 Nenhuma odd válida foi gerada. Por favor, verifique os parâmetros das distribuições ou a odd fixa.")
        st.stop()

    odds_simuladas = np.random.choice(odds_possiveis, size=(num_simulacoes_val, num_apostas_val))

    b_values = odds_simuladas - 1
    
    fatores_kelly = criterio_kelly(p_val, b_values) * fracao_kelly_val
    # Clipe os fatores_kelly para garantir que 1 - f não seja zero ou negativo
    fatores_kelly = np.where(fatores_kelly > 0, fatores_kelly, 0)
    # Limita f para que 1-f > 0, ex: 0.9999 em vez de 1
    fatores_kelly = np.clip(fatores_kelly, 0, 0.99999) 

    resultados_apostas = np.random.rand(num_simulacoes_val, num_apostas_val) < p_val

    retornos = np.where(
        resultados_apostas,
        fatores_kelly * b_values,
        -fatores_kelly
    )

    bankrolls = bankroll_inicial_val * np.cumprod(1 + retornos, axis=1)
    # Adicionar o bankroll inicial na primeira coluna para o histórico completo
    bankrolls_history_with_initial = np.concatenate((np.full((num_simulacoes_val, 1), bankroll_inicial_val), bankrolls), axis=1)
    
    resultados = bankrolls[:, -1]

    # --- Cálculo do Máximo Drawdown para cada simulação ---
    max_drawdowns = calculate_max_drawdowns(bankrolls_history_with_initial)

    bankrolls_maior_inicial = np.sum(resultados > bankroll_inicial_val)

    # --- CÁLCULO DA LINHA DE PROJEÇÃO BASEADA NO CRESCIMENTO LOGARÍTMICO DE KELLY (G) ---
    # Calcular a odd média ponderada (se houver distribuição) ou usar a odd fixa
    if odd_fixa_val is not None:
        avg_odd = odd_fixa_val
    else:
        # Ponderar as odds dentro de cada distribuição e depois ponderar as distribuições
        total_odds_weighted = 0
        total_weight_for_avg_odd = 0
        for pct, i, f, s in distribuicoes_val:
            if pct > 0: # Apenas se o peso for positivo
                segment_odds = arange_inclusivo(i, f, s)
                # Considera que cada odd dentro do segmento tem peso igual para calcular a média do segmento
                avg_segment_odd = np.mean(segment_odds)
                total_odds_weighted += avg_segment_odd * (pct / sum(d[0] for d in distribuicoes_val))
                total_weight_for_avg_odd += (pct / sum(d[0] for d in distribuicoes_val))
        
        avg_odd = total_odds_weighted / total_weight_for_avg_odd if total_weight_for_avg_odd > 0 else 1.01 # Fallback
        
    b_expected_for_G = avg_odd - 1
    
    # Calcular a fração de Kelly aplicada para o cálculo do G
    # Usar np.clip para garantir que o valor esteja entre 0 e 0.99999 (para evitar log(0) ou log(negativo))
    f_kelly_aplicado_for_G = criterio_kelly(p_val, b_expected_for_G) * fracao_kelly_val
    f_kelly_aplicado_for_G = np.clip(f_kelly_aplicado_for_G, 1e-9, 0.99999) # Clip para evitar problemas com log

    # Calcular o crescimento logarítmico esperado (G) por aposta
    # Tratar o caso onde (1 - f_kelly_aplicado_for_G) pode ser muito próximo de zero
    if (1 - f_kelly_aplicado_for_G) <= 1e-9: # Se f é quase 1, o log(1-f) tende a -inf
        G_por_aposta = p_val * np.log(1 + f_kelly_aplicado_for_G * b_expected_for_G)
        if b_expected_for_G <= 0: # Se não há vantagem, G deve ser negativo ou zero
             G_por_aposta = -np.inf # Ou um valor muito pequeno
    else:
        G_por_aposta = p_val * np.log(1 + f_kelly_aplicado_for_G * b_expected_for_G) + \
                       (1 - p_val) * np.log(1 - f_kelly_aplicado_for_G)

    # Criar a linha de projeção do bankroll usando a fórmula de crescimento logarítmico
    kelly_log_growth_path = bankroll_inicial_val * np.exp(np.arange(0, num_apostas_val + 1) * G_por_aposta)

    # --- Estatísticas ---
    taxas_crescimento = np.array(resultados) / bankroll_inicial_val

    ic_inferior_95 = np.percentile(resultados, 2.5)
    ic_superior_95 = np.percentile(resultados, 97.5)
    ic_inferior_68 = np.percentile(resultados, 16)
    ic_superior_68 = np.percentile(resultados, 84)
    mediana = np.median(resultados)

    percentil_5 = np.percentile(resultados, 5)
    percentil_25 = np.percentile(resultados, 25)
    percentil_75 = np.percentile(resultados, 75)
    percentil_95 = np.percentile(resultados, 95)

    valor_minimo = np.min(resultados)
    valor_maximo = np.max(resultados)

    bankrolls_maior_inicial = np.sum(resultados > bankroll_inicial_val)

    chance_bankroll_maior = (bankrolls_maior_inicial / num_simulacoes_val) * 100

    percent_abaixo_inicial = (np.sum(resultados < bankroll_inicial_val) / num_simulacoes_val) * 100
    percent_acima_inicial = (np.sum(resultados >= bankroll_inicial_val) / num_simulacoes_val) * 100

    # Drawdown Metrics
    median_drawdown = np.median(max_drawdowns)
    chance_exceed_drawdown_limit = (np.sum(max_drawdowns >= drawdown_limite_val) / num_simulacoes_val) * 100

    # --- CÁLCULO DA PROJEÇÃO DE APOSTAS PARA AUMENTO DE SALDO X% ---
    num_apostas_para_alvo = "N/A"
    if G_por_aposta > 1e-9: # Deve ser significativamente maior que zero para um crescimento real
        try:
            target_multiplier = 1 + (target_growth_percentage / 100)
            if target_multiplier > 0: # Para evitar log(0) ou log(negativo)
                num_apostas_para_alvo = np.log(target_multiplier) / G_por_aposta
                num_apostas_para_alvo = f"{num_apostas_para_alvo:,.0f} apostas"
            else:
                num_apostas_para_alvo = "Alvo irrealista"
        except Exception:
            num_apostas_para_alvo = "Não calculável"
    else:
        num_apostas_para_alvo = "Infinito (Vantagem nula/negativa)"


    col1, col2, col3, col4 = st.columns(4)

    # Mediana
    with col1:
        if bankroll_inicial_val > 0:
            percentual_crescimento = ((mediana - bankroll_inicial_val) / bankroll_inicial_val) * 100
        else:
            percentual_crescimento = 0
        
        delta_class = "metric-delta-normal" if percentual_crescimento >= 0 else "metric-delta-inverse"
        arrow_html = "&#x2191;" if percentual_crescimento >= 0 else "&#x2193;" # Seta para cima/baixo Unicode
        
        # Determine icon based on delta class
        icon_html = "💰" if percentual_crescimento >= 0 else "📉"


        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">{icon_html} Mediana do Bankroll Final</div>
                <div class="metric-value">R$ {mediana:,.2f}</div>
                <div class="{delta_class} metric-delta">
                    <span class="arrow-icon">{arrow_html}</span>
                    {percentual_crescimento:,.2f}%
                </div>
            </div>
            """)
    
    # Chance de Lucro
    with col2:
        delta_class = "metric-delta-normal" if chance_bankroll_maior >= 50 else "metric-delta-inverse"
        icon_html = "✅" if chance_bankroll_maior >= 50 else "❌"

        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">{icon_html} Bankroll Final > Inicial</div>
                <div class="metric-value">{chance_bankroll_maior:.2f}%</div>
                <div class="metric-delta {delta_class}"></div>
            </div>
            """)
    
    # Valor Mínimo (Pior Cenário)
    with col3:
        if bankroll_inicial_val > 0:
            percentual_pior_cenario = ((valor_minimo - bankroll_inicial_val) / bankroll_inicial_val) * 100
        else:
            percentual_pior_cenario = 0

        delta_class = "metric-delta-normal" if percentual_pior_cenario >= 0 else "metric-delta-inverse"
        arrow_pior_cenario = "&#x2191;" if percentual_pior_cenario >= 0 else "&#x2193;"
        icon_html = "😢" if percentual_pior_cenario < 0 else "👍" # Icon for worst case

        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">{icon_html} Pior Cenário (Mínimo)</div>
                <div class="metric-value">R$ {valor_minimo:,.2f}</div>
                <div class="{delta_class} metric-delta">
                    <span class="arrow-icon">{arrow_pior_cenario}</span>
                    {percentual_pior_cenario:,.2f}%
                </div>
            </div>
            """)
    
    # Valor Máximo (Melhor Cenário)
    with col4:
        if bankroll_inicial_val > 0:
            percentual_melhor_cenario = ((valor_maximo - bankroll_inicial_val) / bankroll_inicial_val) * 100
        else:
            percentual_melhor_cenario = 0

        delta_class = "metric-delta-normal" if percentual_melhor_cenario >= 0 else "metric-delta-inverse"
        arrow_html = "&#x2191;" if percentual_melhor_cenario >= 0 else "&#x2193;"
        icon_html = "🤩" if percentual_melhor_cenario > 0 else "😐" # Icon for best case

        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">{icon_html} Melhor Cenário (Máximo)</div>
                <div class="metric-value">R$ {valor_maximo:,.2f}</div>
                <div class="{delta_class} metric-delta">
                    <span class="arrow-icon">{arrow_html}</span>
                    {percentual_melhor_cenario:,.2f}%
                </div>
            </div>
            """)
    
    # Colunas para as métricas restantes, removendo a coluna de projeção de crescimento
    col_seq_derrotas, col_esperanca_k, col_median_dd, col_chance_dd = st.columns(4) 

    with col_seq_derrotas:
        # Maior Sequência Esperada de Derrotas (continua usando o cálculo original)
        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">💀 Maior Sequência Esperada de Bad Runs</div>
                <div class="metric-value">{(f"{maior_sequencia_derrotas} apostas" if maior_sequencia_derrotas is not None else "Não calculado")}</div>
            </div>
            """)
    
    with col_esperanca_k:
        # Quantidade Esperada de Perder K Consecutivas (agora usa o 'k_derrotas_input')
        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">🔭 Expectativa de {k_derrotas_input} Derrotas Consecutivas</div>
                <div class="metric-value">{(f"{esperanca_perder_k_consecutivas:.2f} vezes" if esperanca_perder_k_consecutivas is not None else "Não calculado")}</div>
            </div>
            """)
            
    with col_median_dd:
        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">📉 Mediana do Drawdown Máximo</div>
                <div class="metric-value">{median_drawdown:,.2f}%</div>
            </div>
            """)

    with col_chance_dd:
        delta_class = "metric-delta-inverse" if chance_exceed_drawdown_limit > 0 else "metric-delta-normal"
        st.html(f"""
            <div class="summary-box">
                <div class="metric-label">⚠️ Chance de Drawdown ≥ {drawdown_limite_val:.1f}%</div>
                <div class="metric-value">{chance_exceed_drawdown_limit:.2f}%</div>
                <div class="metric-delta {delta_class}"></div>
            </div>
            """)
    
    # Nova exibição discreta para a projeção de apostas para o alvo
    st.markdown(
        f"<div class='growth-projection-box'>"
        f"🎯 Para um aumento de saldo de <strong style='color:#4CAF50;'>{target_growth_percentage}%</strong>, são esperadas aproximadamente <strong style='color:#1E90FF;'>{num_apostas_para_alvo}</strong>."
        f"<br><span>O crescimento percentual esperado por aposta é de <strong style='color:#FF5733;'>{G_por_aposta * 100:.2f}%</strong>.</span>"
        f"</div>", 
        unsafe_allow_html=True
    )
           
    st.markdown("---")

# --- GRÁFICO 2 - Caminhos do Bankroll ao Longo das Apostas (COM SLIDER) ---
    with st.expander("⏳ Projeção do Bankroll ao Longo das Apostas"):
        st.write("Visualização da linha mediana e do intervalo interquartil das simulações, mostrando a evolução do bankroll ao longo do número de apostas. A linha roxa representa a projeção de crescimento do bankroll baseada na fórmula de Kelly-Log. Use o slider para ver os valores das linhas em cada aposta.")

        fig_path = go.Figure()

        lower_bound_iqr = np.percentile(bankrolls_history_with_initial, 25, axis=0)
        upper_bound_iqr = np.percentile(bankrolls_history_with_initial, 75, axis=0)

        fig_path.add_trace(go.Scatter(
            x=np.arange(0, num_apostas_val + 1),
            y=upper_bound_iqr,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='none',
            name='Intervalo Interquartil Superior'
        ))
        fig_path.add_trace(go.Scatter(
            x=np.arange(0, num_apostas_val + 1),
            y=lower_bound_iqr,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(192,192,192,0.7)',
            showlegend=True,
            name='Intervalo Interquartil(IQR)',
            hoverinfo='none'
        ))

        median_path = np.median(bankrolls_history_with_initial, axis=0)
        fig_path.add_trace(go.Scatter(
            x=np.arange(0, num_apostas_val + 1),
            y=median_path,
            mode='lines',
            name='Mediana dos Bankrolls',
            line=dict(color='darkblue', width=3),
            hoverinfo='y',
            hovertemplate='Mediana: %{y:,.2f}<extra></extra>'
        ))

        steps = []
        for i in range(num_apostas_val + 1):
            mediana_val_at_i = median_path[i]
            kelly_val_at_i = kelly_log_growth_path[i]
            ic_upper_val_at_i = upper_bound_iqr[i]
            ic_lower_val_at_i = lower_bound_iqr[i]

            step = dict(
                method="relayout",
                args=[{
                    "shapes": [
                        dict(
                            type="line",
                            x0=i,
                            y0=0,
                            x1=i,
                            y1=1,
                            xref="x",
                            yref="paper",
                            line=dict(color="red", dash="dash", width=2)
                        )
                    ],
                    "annotations": [
                        dict(
                            x=i,
                            y=median_path[i],
                            xref="x",
                            yref="y",
                            text=(
                                f'<b>Aposta: {i}</b><br>'
                                f'IQR (Max): R$ {ic_upper_val_at_i:,.2f}<br>'
                                f'Mediana: R$ {mediana_val_at_i:,.2f}<br>'
                                f'IQR (Min): R$ {ic_lower_val_at_i:,.2f}'
                            ),
                            showarrow=True,
                            arrowhead=2,
                            ax=50,
                            ay=-50,
                            bgcolor="rgba(255, 255, 255, 0.4)",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4,
                            font=dict(color="black", size=12),
                            align="left"
                        )
                    ]
                }],
                label=str(i)
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Aposta: "},
            pad={"t": 50},
            steps=steps,
            x=0.05,
            len=0.9
        )]

        fig_path.update_layout(
            title=dict(text='Caminhos do Bankroll ao Longo das Apostas com Projeções e Interatividade', font=dict(size=20, color='black', weight='normal')),
            xaxis_title=dict(text='Número da Aposta', font=dict(size=16, color='black', weight='normal')),
            yaxis_title=dict(text='Bankroll', font=dict(size=16, color='black', weight='normal')),
            xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            yaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=950,
            height=400,
            showlegend=True,
            sliders=sliders
        )
        st.plotly_chart(fig_path, use_container_width=True)

    st.markdown("---")

    with st.expander("📜 Intervalos de Confiança e Percentis"):
        # Criar dados para a tabela com os novos formatos de IC
        data_percentis = {
            "Métrica": ["IC 68%", "IC 95%", 
                        "Percentil 5%", "Percentil 25%", "Percentil 75%", "Percentil 95%"],
            "Valor": [
                f"R$ {ic_inferior_68:,.2f} - R$ {ic_superior_68:,.2f}",
                f"R$ {ic_inferior_95:,.2f} - R$ {ic_superior_95:,.2f}",
                f"R$ {percentil_5:,.2f}",
                f"R$ {percentil_25:,.2f}",
                f"R$ {percentil_75:,.2f}",
                f"R$ {percentil_95:,.2f}"
            ],
            "Descrição": [
                "68% das simulações ficaram dentro deste intervalo.",
                "95% das simulações ficaram dentro deste intervalo.",
                "5% das simulações ficaram abaixo deste valor (risco).",
                "25% das simulações ficaram abaixo deste valor.",
                "75% das simulações ficaram abaixo deste valor.",
                "95% das simulações ficaram abaixo deste valor (potencial de lucro)."
            ]
        }
        
        df_percentis = pd.DataFrame(data_percentis)

        # Função para colorir as células baseada no valor (apenas para valores únicos)
        def color_values(val):
            try:
                # Verificar se é um intervalo (contém "-")
                if '-' in val:
                    # Extrair ambos os valores do intervalo
                    valores = val.replace('R$', '').replace(',', '').split('-')
                    valor_min = float(valores[0].strip())
                    valor_max = float(valores[1].strip())
                    
                    # Colorir baseado na relação com o bankroll inicial
                    if valor_min > bankroll_inicial_val and valor_max > bankroll_inicial_val:
                        color = 'background-color: #d4edda; color: #155724;'  # Verde - ambos acima
                    elif valor_min < bankroll_inicial_val and valor_max < bankroll_inicial_val:
                        color = 'background-color: #f8d7da; color: #721c24;'  # Vermelho - ambos abaixo
                    else:
                        color = 'background-color: #ffeeba; color: #856404;'  # Amarelo - intervalo cruza o inicial
                    return color
                else:
                    # Para valores únicos
                    num_val = float(val.replace('R$', '').replace(',', '').strip())
                    if num_val > bankroll_inicial_val:
                        color = 'background-color: #d4edda; color: #155724;'
                    elif num_val < bankroll_inicial_val:
                        color = 'background-color: #f8d7da; color: #721c24;'
                    else:
                        color = 'background-color: #ffeeba; color: #856404;'
                    return color
            except ValueError:
                return ''

        st.dataframe(df_percentis.style.applymap(color_values, subset=['Valor']), hide_index=True, use_container_width=True)


    st.markdown("---")

# --- NOVO GRÁFICO - Distribuição da Taxa de Crescimento com KDE e Centralização ---
    with st.expander("〽️ Distribuição da Taxa de Crescimento do Bankroll (Curva de Densidade)"):
        st.write("Este gráfico exibe a densidade de probabilidade das taxas de crescimento do bankroll. O valor **1.0** representa o bankroll inicial, com valores acima indicando lucro e valores abaixo indicando prejuízo. O eixo X está centralizado em 1.0 para melhor visualização do ganho/perda relativo.")

        fig_taxa, ax_taxa = plt.subplots(figsize=(12, 7), facecolor='white')  

    # Primeiro, plote o KDE para obter os dados
        sns.kdeplot(
            taxas_crescimento,
            fill=False,
            color='skyblue',
            ax=ax_taxa,
            linewidth=0
        )

        ax_taxa.set_facecolor('white')
        for spine in ax_taxa.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgray')
    
        ax_taxa.tick_params(axis='x', colors='black', labelsize=12)  
        ax_taxa.tick_params(axis='y', colors='black', labelsize=12)
    
        for label in ax_taxa.get_xticklabels():
            label.set_fontweight('normal')
        for label in ax_taxa.get_yticklabels():
            label.set_fontweight('normal')

        ax_taxa.xaxis.label.set_color('black')
        ax_taxa.yaxis.label.set_color('black')
        ax_taxa.title.set_color('black')  
        ax_taxa.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5, color='darkgray', alpha=0.5)

        kde_line_taxa = [line for line in ax_taxa.lines if isinstance(line, plt.Line2D)][0]
        x_data_taxa = kde_line_taxa.get_xdata()
        y_data_taxa = kde_line_taxa.get_ydata()
    
    # Limpar o KDE original
        ax_taxa.clear()
    
    # Reconfigurar o gráfico após clear
        ax_taxa.set_facecolor('white')
        for spine in ax_taxa.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgray')
    
        ax_taxa.tick_params(axis='x', colors='black', labelsize=12)  
        ax_taxa.tick_params(axis='y', colors='black', labelsize=12)
        ax_taxa.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5, color='darkgray', alpha=0.5)
    
    # SOLUÇÃO ALTERNATIVA: Preencher manualmente sem gaps
    # Encontrar o índice mais próximo de 1.0
        idx_1 = np.argmin(np.abs(x_data_taxa - 1.0))
        

    # Percentuais de perda e lucro
        percent_perda = (np.sum(taxas_crescimento < 1.0) / len(taxas_crescimento)) * 100
        percent_lucro = (np.sum(taxas_crescimento >= 1.0) / len(taxas_crescimento)) * 100

    
    # Preencher área vermelha (< 1.0)
        ax_taxa.fill_between(x_data_taxa[:idx_1+1], y_data_taxa[:idx_1+1], 0, 
                            color='red', alpha=0.3, label=f'Perda ({percent_perda:.2f}%)')
    
    # Preencher área verde (> 1.0)
        ax_taxa.fill_between(x_data_taxa[idx_1:], y_data_taxa[idx_1:], 0, 
                            color='green', alpha=0.3, label=f'Lucro ({percent_lucro:.2f}%)')
    
    # Plotar a linha KDE com efeito neon
        cor_neon_taxa = '#4169E1'
        for lw, alpha in zip([15, 10, 6], [0.05, 0.1, 0.3]):
            ax_taxa.plot(x_data_taxa, y_data_taxa, color=cor_neon_taxa, linewidth=lw, alpha=alpha, zorder=5)
        ax_taxa.plot(x_data_taxa, y_data_taxa, color=cor_neon_taxa, linewidth=3, alpha=1, zorder=6, label='Densidade KDE')

        max_diff = max(abs(x_data_taxa.min() - 1.0), abs(x_data_taxa.max() - 1.0))
        ax_taxa.set_xlim(1.0 - max_diff * 1.1, 1.0 + max_diff * 1.1)

    # Adicionar linhas de referência
        mediana_taxa = np.median(taxas_crescimento)
        ic_inferior_95_taxa = np.percentile(taxas_crescimento, 2.5)
        ic_superior_95_taxa = np.percentile(taxas_crescimento, 97.5)
        ic_inferior_68_taxa = np.percentile(taxas_crescimento, 16)
        ic_superior_68_taxa = np.percentile(taxas_crescimento, 84)

        ax_taxa.axvline(mediana_taxa, color='dodgerblue', linestyle='-', linewidth=2, label=f'Mediana ({mediana_taxa:.2f})')
        ax_taxa.axvline(ic_inferior_68_taxa, color='darkorange', linestyle='dotted', linewidth=2)
        ax_taxa.axvline(ic_superior_68_taxa, color='darkorange', linestyle='dotted', linewidth=2, 
                    label=f'IC 68% ({ic_inferior_68_taxa:.2f} - {ic_superior_68_taxa:.2f})')
        ax_taxa.axvline(ic_inferior_95_taxa, color='limegreen', linestyle='dashdot', linewidth=2)
        ax_taxa.axvline(ic_superior_95_taxa, color='limegreen', linestyle='dashdot', linewidth=2, 
                    label=f'IC 95% ({ic_inferior_95_taxa:.2f} - {ic_superior_95_taxa:.2f})')
    
        ax_taxa.set_title('Distribuição da Taxa de Crescimento do Bankroll (Efeito Neon e Eixo Centralizado)', fontsize=18, weight='normal', color='black')
        ax_taxa.set_xlabel('Taxa de Crescimento (1.0 = Bankroll Inicial)', fontsize=16, weight='normal', color='black')
        ax_taxa.set_ylabel('Densidade', fontsize=16, weight='normal', color='black')
        ax_taxa.legend()
        plt.tight_layout()
        st.pyplot(fig_taxa)

    st.markdown("---")



    # --- Gráfico 1 - KDE com Neon/Glow + Sombreamento colorido ---
    with st.expander("📊 Distribuição do Bankroll Final"):
        st.write("O gráfico abaixo mostra a densidade de probabilidade do bankroll final, destacando as áreas de lucro (verde) e prejuízo (vermelho) em relação ao bankroll inicial.")

        fig_kde, ax_kde = plt.subplots(figsize=(12, 7), facecolor='white')  

        sns.histplot(
            resultados,
            bins=100,
            kde=True,
            color='skyblue',
            stat='density',
            edgecolor='black',
            alpha=0.6,
            ax=ax_kde
        )

        ax_kde.set_facecolor('white')  
        for spine in ax_kde.spines.values():
            spine.set_visible(True)
            spine.set_color('darkgray')
        
        ax_kde.tick_params(axis='x', colors='black', labelsize=12)  
        ax_kde.tick_params(axis='y', colors='black', labelsize=12)
        
        for label in ax_kde.get_xticklabels():
            label.set_fontweight('normal')
        for label in ax_kde.get_yticklabels():
            label.set_fontweight('normal')

        ax_kde.xaxis.label.set_color('black')
        ax_kde.yaxis.label.set_color('black')
        ax_kde.title.set_color('black')  
        ax_kde.grid(visible=True, which='both', axis='both', linestyle='--', linewidth=0.5, color='darkgray', alpha=0.5)

        kde_line = [line for line in ax_kde.lines if isinstance(line, plt.Line2D)][0]
        x_data = kde_line.get_xdata()
        y_data = kde_line.get_ydata()
        kde_line.remove()

        ax_kde.fill_between(x_data, y_data, 0, where=(x_data < bankroll_inicial_val), color='red', alpha=0.3, label=f'Abaixo do Bankroll Inicial ({percent_abaixo_inicial:.2f}%)')
        ax_kde.fill_between(x_data, y_data, 0, where=(x_data >= bankroll_inicial_val), color='green', alpha=0.3, label=f'Acima do Bankroll Inicial ({percent_acima_inicial:.2f}%)')

        cor_neon = '#4169E1'
        for lw, alpha in zip([15, 10, 6], [0.05, 0.1, 0.3]):
            ax_kde.plot(x_data, y_data, color=cor_neon, linewidth=lw, alpha=alpha, zorder=5)
        ax_kde.plot(x_data, y_data, color=cor_neon, linewidth=3, alpha=1, zorder=6, label='Densidade KDE')

        ax_kde.axvline(mediana, color='dodgerblue', linestyle='-', linewidth=2, label=f'Mediana ({mediana:.2f})')
        ax_kde.axvline(ic_inferior_68, color='darkorange', linestyle='dotted', linewidth=2)
        ax_kde.axvline(ic_superior_68, color='darkorange', linestyle='dotted', linewidth=2, label=f'IC 68% ({ic_inferior_68:,.2f} - {ic_superior_68:,.2f})')
        ax_kde.axvline(ic_inferior_95, color='limegreen', linestyle='dashdot', linewidth=2)
        ax_kde.axvline(ic_superior_95, color='limegreen', linestyle='dashdot', linewidth=2, label=f'IC 95% ({ic_inferior_95:,.2f} - {ic_superior_95:,.2f})')
        ax_kde.set_title('Distribuição do Bankroll Final com Densidade KDE (Efeito Neon)', fontsize=18, weight='normal', color='black')
        ax_kde.set_xlabel('Bankroll Final', fontsize=16, weight='normal', color='black')
        ax_kde.set_ylabel('Densidade', fontsize=16, weight='normal', color='black')
        ax_kde.legend()
        plt.tight_layout()
        st.pyplot(fig_kde)

    st.markdown("---")



    # --- NOVO GRÁFICO - Histograma de Drawdowns ---
    with st.expander("📈 Distribuição dos Máximos Drawdowns"):
        st.write("O histograma abaixo mostra a distribuição dos drawdowns máximos observados em cada simulação. As cores indicam diferentes níveis de risco:")
        st.markdown(f"- <span style='color:#90EE90'>**Verde**</span>: Drawdown < {drawdown_limite_val:.1f}% (Considerado aceitável)", unsafe_allow_html=True)
        st.markdown(f"- <span style='color:#FF8C00'>**Laranja**</span>: Drawdown ≥ {drawdown_limite_val:.1f}% e < 50% (Atenção)", unsafe_allow_html=True)
        st.markdown(f"- <span style='color:red'>**Vermelho**</span>: Drawdown ≥ 50% (Risco elevado)", unsafe_allow_html=True)

        fig_dd_hist_colored = go.Figure()

        num_bins = 50
        max_dd_for_bins = max(100, max_drawdowns.max() + 5)
        bin_size = max_dd_for_bins / num_bins

        drawdowns_green = max_drawdowns[max_drawdowns < drawdown_limite_val]
        drawdowns_orange = max_drawdowns[(max_drawdowns >= drawdown_limite_val) & (max_drawdowns < 50)]
        drawdowns_red = max_drawdowns[max_drawdowns >= 50]

        if len(drawdowns_green) > 0:
            fig_dd_hist_colored.add_trace(go.Histogram(
                x=drawdowns_green,
                xbins=dict(start=0, end=max_dd_for_bins, size=bin_size),
                marker_color='#90EE90',
                opacity=0.7,
                name=f'Drawdown < {drawdown_limite_val:.1f}%',
                marker_line_width=1,
                marker_line_color="black"
            ))
        if len(drawdowns_orange) > 0:
            fig_dd_hist_colored.add_trace(go.Histogram(
                x=drawdowns_orange,
                xbins=dict(start=0, end=max_dd_for_bins, size=bin_size),
                marker_color='#FF8C00',
                opacity=0.7,
                name=f'{drawdown_limite_val:.1f}% ≤ Drawdown < 50%',
                marker_line_width=1,
                marker_line_color="black"
            ))
        if len(drawdowns_red) > 0:
            fig_dd_hist_colored.add_trace(go.Histogram(
                x=drawdowns_red,
                xbins=dict(start=0, end=max_dd_for_bins, size=bin_size),
                marker_color='red',
                opacity=0.7,
                name='Drawdown ≥ 50%',
                marker_line_width=1,
                marker_line_color="black"
            ))

        fig_dd_hist_colored.update_layout(
            title=dict(text='Distribuição dos Máximos Drawdowns', font=dict(size=20, color='black', weight='normal')),
            xaxis_title=dict(text='Máximo Drawdown (%)', font=dict(size=16, color='black', weight='normal')),
            yaxis_title=dict(text='Frequência', font=dict(size=16, color='black', weight='normal')),
            xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            yaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=950,
            height=600,
            barmode='overlay',
            bargap=0.1,
            showlegend=True
        )

        fig_dd_hist_colored.add_vline(
            x=median_drawdown,
            line_dash="dash",
            line_color="purple",
            annotation_text=f"Mediana: {median_drawdown:.2f}%",
            annotation_position="top right",
            annotation_font_color="purple"
        )
        st.plotly_chart(fig_dd_hist_colored, use_container_width=True)

    st.markdown("---")

    # --- NOVO GRÁFICO - CDF para Drawdowns ---
    with st.expander("〽 CDF do Máximo Drawdown"):
        st.write("A CDF (Função de Distribuição Cumulativa) mostra a probabilidade de o drawdown máximo ser menor ou igual a um determinado valor. Isso ajuda a entender o risco de drawdown. Use o slider para explorar diferentes valores de drawdown.")

        drawdowns_ordenados = np.sort(max_drawdowns)
        cdf_dd = np.arange(1, len(drawdowns_ordenados) + 1) / len(drawdowns_ordenados)

        fig_cdf_dd = go.Figure()

        camadas_glow_dd = [
            {'width': 25, 'opacity': 0.015},
            {'width': 18, 'opacity': 0.03},
            {'width': 12, 'opacity': 0.06},
            {'width': 8, 'opacity': 0.12},
            {'width': 5, 'opacity': 0.2},
        ]
        cor_neon_dd = '#FF4500'

        for camada in camadas_glow_dd:
            fig_cdf_dd.add_trace(go.Scatter(
                x=drawdowns_ordenados,
                y=cdf_dd,
                mode='lines',
                line=dict(color=cor_neon_dd, width=camada['width']),
                opacity=camada['opacity'],
                hoverinfo='skip',
                showlegend=False
            ))

        fig_cdf_dd.add_trace(go.Scatter(
            x=drawdowns_ordenados,
            y=cdf_dd,
            mode='lines',
            line=dict(color=cor_neon_dd, width=2),
            name='CDF do Drawdown',
            hovertemplate='Drawdown: %{x:.2f}%<br>Probabilidade acumulada ≤: %{y:.2%}<extra></extra>'
        ))

        fig_cdf_dd.update_layout(
            title=dict(text='CDF do Máximo Drawdown (Efeito Neon e Slider Interativo)', font=dict(size=20, color='black', weight='normal')),
            xaxis_title=dict(text='Máximo Drawdown (%)', font=dict(size=16, color='black', weight='normal')),
            yaxis_title=dict(text='Probabilidade Acumulada', font=dict(size=16, color='black', weight='normal')),
            xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            yaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=950,
            height=600,
            showlegend=True
        )

        steps_dd = []
        if len(drawdowns_ordenados) > 1:
            min_dd_slider = max(0, drawdowns_ordenados.min() - (drawdowns_ordenados.max() - drawdowns_ordenados.min()) * 0.1)
            max_dd_slider = drawdowns_ordenados.max() + (drawdowns_ordenados.max() - drawdowns_ordenados.min()) * 0.1
            step_dd_size = (max_dd_slider - min_dd_slider) / 100
        else:
            min_dd_slider = 0
            max_dd_slider = 100
            step_dd_size = 1

        slider_vals_dd = np.linspace(min_dd_slider, max_dd_slider, 100)
        
        for i, valor in enumerate(slider_vals_dd):
            prob = np.interp(valor, drawdowns_ordenados, cdf_dd)
            step = dict(
                method="relayout",
                args=[{
                    "shapes": [
                        dict(
                            type="line",
                            x0=valor,
                            y0=0,
                            x1=valor,
                            y1=prob,
                            line=dict(color="red", dash="dash", width=2)
                        ),
                        dict(
                            type="line",
                            x0=0,
                            y0=prob,
                            x1=valor,
                            y1=prob,
                            line=dict(color="grey", dash="dot", width=1)
                        )
                    ],
                    "annotations": [
                        dict(
                            x=valor,
                            y=prob,
                            xref="x",
                            yref="y",
                            text=f'Drawdown: {valor:.2f}%<br>Prob: {prob:.2%}',
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-30,
                            font=dict(color="black", size=12),
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4
                        )
                    ]
                }],
                label=f' {valor:.2f}%'
            )
            steps_dd.append(step)

        sliders_dd = [dict(
            active=0,
            currentvalue={"prefix": "Drawdown: "},
            pad={"t": 60},
            steps=steps_dd,
            x=0.05,
            len=0.9
        )]

        fig_cdf_dd.update_layout(
            sliders=sliders_dd
        )

        st.plotly_chart(fig_cdf_dd, use_container_width=True)

    st.markdown("---")


    # --- GRÁFICO - CDF da Taxa de Crescimento ---
    with st.expander("↗️ CDF da Taxa de Crescimento do Bankroll"):
        st.write("A CDF (Função de Distribuição Cumulativa) mostra a probabilidade de a taxa de crescimento ser menor ou igual a um determinado valor. Isso ajuda a entender a probabilidade de atingir ou exceder um certo retorno. Use o slider para explorar diferentes valores de taxa de crescimento.")

        taxas_ordenadas = np.sort(taxas_crescimento)
        cdf_taxa = np.arange(1, len(taxas_ordenadas) + 1) / len(taxas_ordenadas)

        fig_cdf_taxa = go.Figure()

        camadas_glow = [
            {'width': 25, 'opacity': 0.015},
            {'width': 18, 'opacity': 0.03},
            {'width': 12, 'opacity': 0.06},
            {'width': 8, 'opacity': 0.12},
            {'width': 5, 'opacity': 0.2},
        ]
        cor_neon_taxa_cdf = '#00FFFF'

        for camada in camadas_glow:
            fig_cdf_taxa.add_trace(go.Scatter(
                x=taxas_ordenadas,
                y=cdf_taxa,
                mode='lines',
                line=dict(color=cor_neon_taxa_cdf, width=camada['width']),
                opacity=camada['opacity'],
                hoverinfo='skip',
                showlegend=False
            ))

        fig_cdf_taxa.add_trace(go.Scatter(
            x=taxas_ordenadas,
            y=cdf_taxa,
            mode='lines',
            line=dict(color=cor_neon_taxa_cdf, width=2),
            name='CDF da Taxa de Crescimento',
            hovertemplate='Taxa de Crescimento: %{x:.2f}<br>Probabilidade acumulada ≤: %{y:.2%}<extra></extra>'
        ))

        fig_cdf_taxa.update_layout(
            title=dict(text='CDF da Taxa de Crescimento do Bankroll (Efeito Neon e Slider Interativo)', font=dict(size=20, color='black', weight='normal')),
            xaxis_title=dict(text='Taxa de Crescimento (1.0 = Bankroll Inicial)', font=dict(size=16, color='black', weight='normal')),
            yaxis_title=dict(text='Probabilidade Acumulada', font=dict(size=16, color='black', weight='normal')),
            xaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            yaxis=dict(showgrid=True, zeroline=False, tickfont=dict(color='black', size=12, weight='normal')),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=950,
            height=600,
            showlegend=True
        )

        steps_taxa_cdf = []
        min_taxa_slider = max(0, taxas_ordenadas.min() - (taxas_ordenadas.max() - taxas_ordenadas.min()) * 0.1)
        max_taxa_slider = taxas_ordenadas.max() + (taxas_ordenadas.max() - taxas_ordenadas.min()) * 0.1
        slider_vals_taxa = np.linspace(min_taxa_slider, max_taxa_slider, 100)

        for i, valor in enumerate(slider_vals_taxa):
            prob = np.interp(valor, taxas_ordenadas, cdf_taxa)
            step = dict(
                method="relayout",
                args=[{
                    "shapes": [
                        dict(
                            type="line",
                            x0=valor,
                            y0=0,
                            x1=valor,
                            y1=prob,
                            line=dict(color="red", dash="dash", width=2)
                        ),
                        dict(
                            type="line",
                            x0=0,
                            y0=prob,
                            x1=valor,
                            y1=prob,
                            line=dict(color="grey", dash="dot", width=1)
                        )
                    ],
                    "annotations": [
                        dict(
                            x=valor,
                            y=prob,
                            xref="x",
                            yref="y",
                            text=f'Taxa: {valor:.2f}<br>Prob: {prob:.2%}',
                            showarrow=True,
                            arrowhead=2,
                            ax=20,
                            ay=-30,
                            font=dict(color="black", size=12),
                            bgcolor="rgba(255, 255, 255, 0.7)",
                            bordercolor="black",
                            borderwidth=1,
                            borderpad=4
                        )
                    ]
                }],
                label=f'{valor:.2f}'
            )
            steps_taxa_cdf.append(step)

        sliders_taxa_cdf = [dict(
            active=0,
            currentvalue={"prefix": "Taxa de Crescimento: "},
            pad={"t": 60},
            steps=steps_taxa_cdf,
            x=0.05,
            len=0.9
        )]

        fig_cdf_taxa.update_layout(
            sliders=sliders_taxa_cdf
        )

        st.plotly_chart(fig_cdf_taxa, use_container_width=True)

    st.markdown("---")
    
    # --- NOVO GRÁFICO: Distribuição de Probabilidade das Porcentagens de Vitória (Com métrica abaixo) ---



    # ✅ NOVO GRÁFICO: Distribuição de Probabilidade das Porcentagens de Vitória
    with st.expander("📉 Distribuição de Probabilidade das Porcentagens de Vitória"):
        st.write("Este gráfico mostra a distribuição de probabilidade das porcentagens de vitória, com base nos parâmetros de Probabilidade de Acerto (p) e Número de Apostas. Inclui a curva da distribuição normal como aproximação.")

        n_binomial = num_apostas_val
        p_binomial = p_val

        vitorias_possiveis = np.arange(0, n_binomial + 1)
        probabilidades_binomial = binom.pmf(vitorias_possiveis, n_binomial, p_binomial)
        porcentagens_vitoria = (vitorias_possiveis / n_binomial) * 100

        mean_wins = n_binomial * p_binomial
        std_dev_wins = math.sqrt(n_binomial * p_binomial * (1 - p_binomial))
        x_normal = np.linspace(max(0, mean_wins - 4 * std_dev_wins), min(n_binomial, mean_wins + 4 * std_dev_wins), 500)
        pdf_normal_wins = norm.pdf(x_normal, mean_wins, std_dev_wins)
        pdf_normal_scaled = pdf_normal_wins * (100 / n_binomial) if n_binomial > 0 else np.zeros_like(pdf_normal_wins)
        porcentagens_normal_curva = (x_normal / n_binomial) * 100 if n_binomial > 0 else np.zeros_like(x_normal)

        mean_percent_wins = p_binomial * 100
        std_dev_percent_wins = (std_dev_wins / n_binomial) * 100 if n_binomial > 0 else 0
        ic_95_inf = mean_percent_wins - 1.96 * std_dev_percent_wins
        ic_95_sup = mean_percent_wins + 1.96 * std_dev_percent_wins

        bar_width = (100 / n_binomial) * 0.7 if n_binomial > 0 else 1

        fig_prob_dist = go.Figure()

        fig_prob_dist.add_trace(go.Bar(
            x=porcentagens_vitoria,
            y=probabilidades_binomial * 100,
            name='Histograma Binomial',
            marker_color='#0066cc',
            opacity=0.7,
            marker_line_width=1,
            marker_line_color="black",
            hoverinfo='x+y',
            hovertemplate='Porcentagem de Vitória: %{x:.2f}%<br>Probabilidade: %{y:.2f}%<extra></extra>',
            width=bar_width
        ))

        fig_prob_dist.add_trace(go.Scatter(
            x=porcentagens_normal_curva,
            y=pdf_normal_scaled * 100,
            mode='lines',
            name='Curva Normal (Aproximação)',
            line=dict(color='#29b6f6', width=3),
            opacity=0.8,
            hovertemplate='Porcentagem de Vitória: %{x:.2f}%<br>Densidade (Escalada): %{y:.2f}%<extra></extra>',
        ))

        x_range_min = max(0, mean_percent_wins - 4 * std_dev_percent_wins)
        x_range_max = min(100, mean_percent_wins + 4 * std_dev_wins)
        padding_x = (x_range_max - x_range_min) * 0.05
        x_range_final_min = max(0, x_range_min - padding_x)
        x_range_final_max = min(100, x_range_max + padding_x)

        info_text = (
            f"<b>Hit Rate Esperado:</b> {mean_percent_wins:.0f}%<br>"
            f"<b>Desvio Padrão:</b> {std_dev_percent_wins:.2f}%<br>"
            f"<b>Intervalo de Confiança (95%):</b> {ic_95_inf:.0f}% a {ic_95_sup:.0f}%"
        )

        fig_prob_dist.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0.05, y0=-0.25, x1=0.95, y1=-0.10,
            line=dict(color="rgba(0,0,0,0.2)", width=1),
            fillcolor="rgba(240,240,240,0.6)",
            layer="below"
        )

        fig_prob_dist.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.24,
            showarrow=False,
            align="center",
            font=dict(size=14, color="black"),
            borderpad=5
        )

        fig_prob_dist.update_layout(
            title=dict(text='Distribuição de Probabilidade das Porcentagens de Vitória',     font=dict(size=20, color='black', weight='normal')),
            xaxis_title=dict(text='Taxa de Vitória', font=dict(size=15, color='black',     weight='normal')),
            yaxis_title=dict(text='Probabilidade (%) / Densidade (Escalada)', font=dict    (size=16, color='black', weight='normal')),
            xaxis=dict(
                showgrid=True,
                zeroline=False,
                tickfont=dict(color='black', size=12, weight='normal'),
                range=[x_range_final_min, x_range_final_max],
                tickmode='linear',
                dtick=max(1, round(100 / n_binomial)),
                title_standoff=5,
                ticksuffix="%"
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                tickfont=dict(color='black', size=12, weight='normal')
            ),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=950,
            height=700,
            barmode='overlay',
            bargap=0.01,
            showlegend=True,
            margin=dict(b=120),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(255, 255, 255, 0.7)',
                bordercolor='Black',
                borderwidth=1
            )
        )

        st.plotly_chart(fig_prob_dist, use_container_width=True)

    st.markdown("---")

# --- Executar e Mostrar Resultados da Mini Simulação Condicional ---
    with st.expander("🔮 Resultados da Projeção Futura Condicional"):
        st.write(f"Simulando {num_simulacoes_mini} novos caminhos a partir da **Aposta {aposta_y_mini_sim}** com **R$ {quantia_x_mini_sim:,.2f}** de bankroll:")

        chance_maior_igual_x_mini, chance_maior_inicial_mini, bankrolls_finais_mini = mini_simulacao_condicional(
            bankroll_inicial_val, # Importante passar o bankroll_inicial_val original
            quantia_x_mini_sim,
            aposta_y_mini_sim,
            num_apostas_val,
            p_val,
            odd_fixa_val,
            distribuicoes_val,
            fracao_kelly_val,
            num_simulacoes_mini
        )

        if chance_maior_igual_x_mini is not None:
            col_mini1, col_mini2 = st.columns(2)
            with col_mini1:
                delta_class = "metric-delta-normal" if chance_maior_igual_x_mini >= 50 else "metric-delta-inverse"
                st.html(f"""
                    <div class="summary-box">
                        <div class="metric-label">Chance de terminar > R$ {quantia_x_mini_sim:,.2f} (Bankroll de Partida da Projeção Futura)</div>
                        <div class="metric-value">{chance_maior_igual_x_mini:.2f}%</div>
                        <div class="metric-delta {delta_class}"></div>
                    </div>
                    """)
            with col_mini2:
                delta_class = "metric-delta-normal" if chance_maior_inicial_mini >= 50 else "metric-delta-inverse"
                st.html(f"""
                    <div class="summary-box">
                        <div class="metric-label">Chance de terminar > Bankroll Inicial Total (R$ {bankroll_inicial_val:,.2f})</div>
                        <div class="metric-value">{chance_maior_inicial_mini:.2f}%</div>
                        <div class="metric-delta {delta_class}"></div>
                    </div>
                    """)
            # Opcional: Visualização da distribuição dos bankrolls finais da mini-simulação
            # Opcional: Visualização da distribuição dos bankrolls finais da mini-simulação
            # 📊 Curva Interativa da Taxa de Crescimento (Plotly)
            st.subheader("Distribuições da Projeção Futura Condicional:")
            
            taxas_crescimento_mini = bankrolls_finais_mini / quantia_x_mini_sim
            taxas_crescimento_mini_ordenadas = np.sort(taxas_crescimento_mini)
            cdf_mini = np.arange(1, len(taxas_crescimento_mini_ordenadas) + 1) / len(taxas_crescimento_mini_ordenadas)

            from scipy.stats import gaussian_kde
            kde_func = gaussian_kde(taxas_crescimento_mini)
            x_kde = np.linspace(max(0, taxas_crescimento_mini.min() * 0.9), taxas_crescimento_mini.max() * 1.1, 500)
            y_kde = kde_func(x_kde)
            max_diff = max(abs(x_kde.min() - 1.0), abs(x_kde.max() - 1.0))
            x_range = [1.0 - max_diff * 1.1, 1.0 + max_diff * 1.1]		

            fig_kde_interativo = go.Figure()

            # Neon-style glow layers
            for lw, opacity in zip([20, 14, 8, 4], [0.05, 0.1, 0.2, 0.4]):
                fig_kde_interativo.add_trace(go.Scatter(
                    x=x_kde,
                    y=y_kde,
                    mode='lines',
                    line=dict(color='gold', width=lw),
                    name='',
                    hoverinfo='skip',
                    showlegend=False,
                    opacity=opacity
                ))

            # Linha principal
            fig_kde_interativo.add_trace(go.Scatter(
                x=x_kde,
                y=y_kde,
                mode='lines',
                line=dict(color='gold', width=2),
                name='Densidade KDE',
                hovertemplate='Taxa: %{x:.3f}<br>Densidade: %{y:.4f}<extra></extra>'
            ))

            # Linha vertical em 1.0 (ponto neutro)
            fig_kde_interativo.add_vline(
                x=1.0,
                line=dict(color='black', dash='dash'),
                annotation_text='Bankroll Inicial (1.0)',
                annotation_position='top left',
                annotation_font_size=12
            )

            # Slider interativo para explorar a CDF + percentil
            steps = []
            valores_slider = np.linspace(x_kde.min(), x_kde.max(), 100)
            for val in valores_slider:
                prob = float(np.interp(val, taxas_crescimento_mini_ordenadas, cdf_mini))
                percentil_aprox = prob * 100

                interpretacao = "🔴 Prejuízo" if val < 1.0 else "🟢 Lucro"
                texto = f'Taxa: {val:.3f}<br>Percentil: {percentil_aprox:.1f}%<br>{interpretacao}'

                step = dict(
                    method="relayout",
                    args=[{
                        "shapes": [
                            dict(
                                type="line",
                                x0=val, x1=val,
                                y0=0, y1=max(y_kde) * 1.05,
                                line=dict(color="red", dash="dot", width=2)
                            )
                        ],
                        "annotations": [
                            dict(
                                x=val,
                                y=max(y_kde) * 0.95,
                                text=texto,
                                showarrow=True,
                                arrowhead=2,
                                ax=30,
                                ay=-40,
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="black",
                                borderwidth=1,
                                font=dict(size=12)
                            )
                        ]
                    }],
                    label=f"{val:.2f}"
                )
                steps.append(step)

            sliders = [dict(
                active=50,
                currentvalue={"prefix": "Taxa: "},
                pad={"t": 60},
                steps=steps
            )]

            fig_kde_interativo.update_layout(
                title='Distribuição da Taxa de Crescimento (KDE Interativo com CDF e Percentil)',
                xaxis_title='Taxa de Crescimento (Bankroll Final / X)',
                yaxis_title='Densidade',
                template='plotly_white',
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600,
                sliders=sliders,
		xaxis=dict(range=x_range)
            )

            
            aba_kde, aba_hist = st.tabs(["📈 Curva KDE Interativa", "📊 Histograma do Bankroll Final"])

            with aba_kde:
                st.write("A curva abaixo mostra a distribuição da taxa de crescimento do bankroll a partir do ponto de reentrada (X). O valor 1.0 representa o bankroll inicial da nova simulação.")
                st.plotly_chart(fig_kde_interativo, use_container_width=True)

            with aba_hist:
                st.write("O histograma abaixo mostra a distribuição dos bankrolls finais da projeção futura condicional, destacando em verde os valores acima do bankroll de partida (X) e em vermelho os valores abaixo.")
                plot_histograma_tricolor_mini_sim(bankrolls_finais_mini, quantia_x_mini_sim, bankroll_inicial_val)
        
    # --- Substitua esta seção no código existente ---