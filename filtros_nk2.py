import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import neurokit2 as nk

# --- 1. Carregamento e Configuração ---
fs = 360  # Frequência de amostragem
try:
    df = pd.read_csv('signal_tables/100_record.csv')
    ecg_signal = df['V5'].values
except (FileNotFoundError, KeyError):
    print("Arquivo não encontrado. Usando ECG simulado.")
    ecg_signal = nk.ecg_simulate(duration=10, sampling_rate=fs, heart_rate=70, noise=0.05)

# --- 2. Encontrar Picos R e Delinear o Sinal ---
_, r_peaks = nk.ecg_peaks(ecg_signal, sampling_rate=fs)
_, waves = nk.ecg_delineate(ecg_signal, r_peaks, sampling_rate=fs, method="dwt")

# --- 3. Visualização Manual e Completa com Matplotlib ---
# Criar o vetor de tempo em segundos
time = np.arange(len(ecg_signal)) / fs

# Dicionário para facilitar o plot de cada tipo de pico
peak_info = {
    'ECG_P_Peaks': ('P', 'blue', 'o'),
    'ECG_Q_Peaks': ('Q', 'green', 'x'),
    'ECG_R_Peaks': ('R', 'red', '^'),
    'ECG_S_Peaks': ('S', 'purple', 'x'),
    'ECG_T_Peaks': ('T', 'orange', 'o')
}
# O dicionário 'waves' não inclui os picos R, então adicionamos manualmente
waves['ECG_R_Peaks'] = r_peaks['ECG_R_Peaks']

plt.figure(figsize=(20, 8))
# Plotar o sinal de ECG base
plt.plot(time, ecg_signal, label='Sinal ECG', color='black', alpha=0.5)

# Iterar sobre cada tipo de pico para plotá-los no MESMO gráfico
for wave_name, (label, color, marker) in peak_info.items():
    # Pegar os índices dos picos (removendo NaNs se houver)
    indices = waves[wave_name]
    indices = [i for i in indices if not np.isnan(i)] # Garantir que não há NaNs
    
    # Plotar cada pico usando o vetor de tempo
    plt.plot(time[indices], ecg_signal[indices], marker, color=color, markersize=8, label=f'Picos {label}', linestyle='None')

# Configurações do gráfico
plt.title("Delineação Completa do ECG (Eixo X em Segundos)")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
#plt.xlim(2, 6)  # Foco em alguns batimentos para melhor visualização
plt.legend()
plt.grid(True)
plt.show()