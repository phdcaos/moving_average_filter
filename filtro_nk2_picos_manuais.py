import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

# =============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO
# =============================================================================

# --- Correção Crítica ---
fs = 360  # Frequência de amostragem CORRETA para '100_record.csv'

try:
    df = pd.read_csv('signal_tables/100_record.csv')
    ecg_signal = df['V5'].values
except (FileNotFoundError, KeyError) as e:
    print(f"Erro: {e}. Usando um sinal de exemplo.")
    ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, noise=0.05)

# --- Limpeza de Baseline com NeuroKit2 (usando o fs correto) ---
ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")

# --- Aplicação do Filtro de Média Móvel para Suavização ---
window_size = 20  # Janela de 11 amostras
s_cleaned = pd.Series(ecg_cleaned)
ecg_smoothed = s_cleaned.rolling(window=window_size, center=True).mean()
ecg_smoothed.fillna(method='bfill', inplace=True) # Preenche NaNs no início
ecg_smoothed.fillna(method='ffill', inplace=True) # Preenche NaNs no fim
ecg_smoothed = ecg_smoothed.values # Converte de volta para array numpy


# =============================================================================
# 2. DETECTOR MANUAL DE PICOS E VALES (Estilo MATLAB)
# =============================================================================

def detectar_extremos_manualmente(sinal):
    """
    Itera pelo sinal e encontra todos os máximos e mínimos locais.
    Um ponto é um pico se for maior que seus vizinhos imediatos.
    Um ponto é um vale se for menor que seus vizinhos imediatos.
    Retorna os ÍNDICES dos picos e vales.
    """
    indices_picos = []
    indices_vales = []

    # Itera do segundo (índice 1) ao penúltimo elemento
    for i in range(1, len(sinal) - 1):
        vizinho_anterior = sinal[i-1]
        ponto_atual = sinal[i]
        vizinho_seguinte = sinal[i+1]

        # Condição para ser um pico (máximo local)
        if ponto_atual > vizinho_anterior and ponto_atual > vizinho_seguinte:
            indices_picos.append(i)
            
        # Condição para ser um vale (mínimo local)
        elif ponto_atual < vizinho_anterior and ponto_atual < vizinho_seguinte:
            indices_vales.append(i)
            
    return indices_picos, indices_vales

# Aplicar a função no sinal suavizado
picos_detectados_indices, vales_detectados_indices = detectar_extremos_manualmente(ecg_smoothed)


# =============================================================================
# 3. VISUALIZAÇÃO DOS RESULTADOS
# =============================================================================

time = np.arange(len(ecg_smoothed)) / fs

plt.figure(figsize=(20, 8))

# Plotar o sinal onde a detecção foi feita
plt.plot(time, ecg_smoothed, label=f'Sinal Suavizado (Média Móvel, Janela={window_size})', color='black', alpha=0.8)

# Marcar os picos e vales encontrados
plt.plot(time[picos_detectados_indices], ecg_smoothed[picos_detectados_indices], 'x', color='red', markersize=8, label='Picos Detectados (Manual)')
plt.plot(time[vales_detectados_indices], ecg_smoothed[vales_detectados_indices], 'o', color='blue', markersize=8, label='Vales Detectados (Manual)')

# Configurações do gráfico
plt.title('Detecção Manual de Picos/Vales Após Média Móvel')
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
#plt.xlim(2, 6) # Foco em 4 segundos para melhor visualização
plt.legend()
plt.grid(True)
plt.show()

print(f"Total de picos positivos encontrados: {len(picos_detectados_indices)}")
print(f"Total de vales (picos negativos) encontrados: {len(vales_detectados_indices)}")