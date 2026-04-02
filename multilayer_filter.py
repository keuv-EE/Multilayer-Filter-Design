import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0

def multilayer_filter():
    # Parâmetros físicos
    c = 3e8  # velocidade da luz no vácuo [m/s]
    eta_0 = 120 * np.pi  # impedância do vácuo [Ω]
    
    lambda_central = 600e-9  # Centro em 600nm
    
    # Definir a estrutura multicamadas
    n_a = 1  # Meio de entrada
    n_H = 3.5  # Meio do filtro com maior índice de refração
    n_L = 1.46  # Meio do filtro com menor índice de refração
    n_b = 1  # Meio de saída
    
    # Vetor índice de refração
    n = np.array([n_a, n_L, n_H, n_L, n_H, n_L, n_H, n_L, n_b])
    
    # Espessura do i-ésimo meio
    l = np.array([0, lambda_central/(4*n_L), lambda_central/(4*n_H),
                  lambda_central/(4*n_L), lambda_central/(2*n_H),
                  lambda_central/(4*n_L), lambda_central/(4*n_H),
                  lambda_central/(4*n_L), 0])
    
    # Número total de meios
    num_meios = len(n)
    # Número de interfaces
    num_interfaces = num_meios - 1
    
    # Faixa de comprimentos de onda para análise [m]
    lambda_min = 300e-9
    lambda_max = 900e-9
    num_lambda = 1000
    lambda_vec = np.linspace(lambda_min, lambda_max, num_lambda)
    
    # Vetores para armazenar a resposta do filtro
    transmission = np.zeros(num_lambda)
    reflection = np.zeros(num_lambda)
    
    # Campo magnético de saída (normalizado)
    H_out = 1  # H'_{M+1},+ no meio b
    
    for idx in range(num_lambda):
        lambda_0 = lambda_vec[idx]
        k_0 = 2 * np.pi / lambda_0
        
        # Calcular impedâncias de cada meio
        eta = eta_0 / n
        
        # Inicializar coeficientes de reflexão
        gamma = np.zeros(num_meios, dtype=complex)
        
        # No último meio (n_b), não há onda refletida
        gamma[num_meios-1] = 0
        
        # Calcular coeficientes de reflexão recursivamente
        for i in range(num_interfaces-1, -1, -1):
            # Coeficientes de reflexão na interface i
            pho_i = (eta[i] - eta[i+1]) / (eta[i] + eta[i+1])
            
            # Fator de fase para a camada i (se tiver espessura finita)
            if l[i] > 0:
                k_i = k_0 * n[i]
                alfa_minus = np.exp(-1j * k_i * l[i])
                
                # Atualizar coeficiente de reflexão considerando a camada
                gamma_i_temp = (gamma[i+1] + pho_i) / (pho_i * gamma[i+1] + 1)
                gamma[i] = gamma_i_temp * alfa_minus**2
            else:
                # Para meios semi-infinitos (a e b)
                gamma[i] = (gamma[i+1] + pho_i) / (pho_i * gamma[i+1] + 1)
        
        # Calcular campos magnéticos em cada meio
        H_plus = np.zeros(num_meios, dtype=complex)
        H_minus = np.zeros(num_meios, dtype=complex)
        
        # Condição de contorno: campo de saída no meio b
        H_plus[num_meios-1] = H_out
        H_minus[num_meios-1] = 0
        
        # Propagação reversa: calcular campos em cada interface
        for i in range(num_interfaces-1, -1, -1):
            pho_i = (eta[i] - eta[i+1]) / (eta[i] + eta[i+1])
            tal_i = 2 * eta[i] / (eta[i] + eta[i+1])
            
            if l[i] > 0:
                # Camada com espessura finita
                k_i = k_0 * n[i]
                alfa_minus = np.exp(-1j * k_i * l[i])
                alfa_plus = np.exp(1j * k_i * l[i])
                
                # Matriz de transferência
                T = np.array([[alfa_plus, pho_i * alfa_plus],
                             [pho_i * alfa_minus, alfa_minus]])
                
                # Campos no meio i
                campos_i = (1/tal_i) * T @ np.array([H_plus[i+1], H_minus[i+1]])
                H_plus[i] = campos_i[0]
                H_minus[i] = campos_i[1]
            else:
                # Para meios semi-infinitos
                T = np.array([[1, pho_i], [pho_i, 1]])
                campos_i = (1/tal_i) * T @ np.array([H_plus[i+1], H_minus[i+1]])
                H_plus[i] = campos_i[0]
                H_minus[i] = campos_i[1]
        
        # Calcular potências
        Pin = (np.abs(H_plus[0])**2) / (2 * np.real(eta[0]))
        Pref = (np.abs(H_minus[0])**2) / (2 * np.real(eta[0]))
        Ptr = (np.abs(H_plus[-1])**2) / (2 * np.real(eta[-1]))
        
        # Coeficiente de transmissão e reflexão
        transmission[idx] = Ptr / Pin
        reflection[idx] = Pref / Pin
    
    # Análise do desempenho do filtro
    lambda_nm = lambda_vec * 1e9
    
    # 1. Transmissão média na banda de 572-630nm
    banda_passagem = (lambda_nm >= 572) & (lambda_nm <= 630)
    transmissao_media_banda = np.mean(transmission[banda_passagem])
    
    # 2. Transmissão máxima fora da banda
    fora_banda = (lambda_nm < 572) | (lambda_nm > 630)
    transmissao_max_fora = np.max(transmission[fora_banda])
    
    # 3. Relação de rejeição em dB
    relacao_rejeicao_dB = 10 * np.log10(transmissao_media_banda / transmissao_max_fora)
    
    # Exibir resultados no console
    print('\n--- DESEMPENHO DO FILTRO ---')
    print(f'Transmissão média na banda 572-630nm: {transmissao_media_banda:.4f} ({transmissao_media_banda*100:.2f}%)')
    print(f'Transmissão máxima fora da banda: {transmissao_max_fora:.4f} ({transmissao_max_fora*100:.2f}%)')
    print(f'Relação de rejeição: {relacao_rejeicao_dB:.2f} dB')
    
    # Transmissão no comprimento de onda central
    idx_central = np.argmin(np.abs(lambda_nm - 600))
    transmissao_central = transmission[idx_central]
    print(f'Transmissão em 600nm: {transmissao_central:.4f} ({transmissao_central*100:.2f}%)')
    
    # Configurar estilo dos gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Plotar resultados - Transmissão em dB
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec * 1e9, 10*np.log10(transmission), 'r-', linewidth=2)
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Transmissão (dB)')
    plt.title('Resposta do Filtro Multicamadas - Modo TM')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plotar resultados - Transmissão linear
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec * 1e9, transmission, 'b-', linewidth=2)
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Transmissão (linear)')
    plt.title('Resposta do Filtro Multicamadas - Modo TM')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plotar resultados - Reflexão em dB
    plt.figure(figsize=(10, 6))
    # Evitar log de zero adicionando um pequeno epsilon
    reflection_db = 10 * np.log10(reflection + 1e-10)
    plt.plot(lambda_vec * 1e9, reflection_db, 'r-', linewidth=2)
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Reflexão (dB)')
    plt.title('Coeficiente de Reflexão do Filtro Multicamadas')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plotar resultados - Reflexão linear
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec * 1e9, reflection, 'purple', linewidth=2)
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Reflexão (linear)')
    plt.title('Coeficiente de Reflexão do Filtro Multicamadas')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plotar transmissão e reflexão juntas para comparação
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec * 1e9, transmission, 'b-', linewidth=2, label='Transmissão')
    plt.plot(lambda_vec * 1e9, reflection, 'r-', linewidth=2, label='Reflexão')
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Coeficiente (linear)')
    plt.title('Transmissão vs Reflexão - Filtro Multicamadas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Plot adicional: Transmissão e Reflexão em dB no mesmo gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(lambda_vec * 1e9, 10*np.log10(transmission), 'b-', linewidth=2, label='Transmissão')
    plt.plot(lambda_vec * 1e9, reflection_db, 'r-', linewidth=2, label='Reflexão')
    plt.xlabel('Comprimento de Onda (nm)')
    plt.ylabel('Coeficiente (dB)')
    plt.title('Transmissão vs Reflexão (dB) - Filtro Multicamadas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Salvar dados
    np.savez('filter_response.npz', 
             lambda_vec=lambda_vec, 
             transmission=transmission, 
             reflection=reflection)
    
    print(f'\nSimulação concluída. Número total de meios: {num_meios}')
    print(f'Número de interfaces: {num_interfaces}')
    print(f'Transmissão máxima: {np.max(transmission):.4f}')
    print(f'Reflexão máxima: {np.max(reflection):.4f}')
    
    # Mostrar todos os gráficos
    plt.show()

if __name__ == "__main__":
    multilayer_filter()