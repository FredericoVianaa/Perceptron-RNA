import numpy as np

# Definição da função de ativação (Sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da Sigmoid para usar no backpropagation
def sigmoid_derivada(x):
    return x * (1 - x)

# Dados de treinamento
treino_inputs = np.array([
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],  
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],  
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]  
])

treino_saidas = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])  

# Inicialização dos pesos e bias
np.random.seed(42)  # Para reprodutibilidade
pesos_input_oculta = np.random.rand(16, 8) - 0.5  
pesos_oculta_saida = np.random.rand(8, 1) - 0.5  
bias_oculta = np.random.rand(1, 8) - 0.5  
bias_saida = np.random.rand(1, 1) - 0.5  

# Hiperparâmetros
taxa_aprendizado = 0.5
epocas = 10000  

# Treinamento com Backpropagation
for epoca in range(epocas):
    # Forward Pass
    camada_oculta = sigmoid(np.dot(treino_inputs, pesos_input_oculta) + bias_oculta)
    saida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida) + bias_saida)

    # Cálculo do erro
    erro_saida = treino_saidas - saida
    delta_saida = erro_saida * sigmoid_derivada(saida)

    erro_oculta = delta_saida.dot(pesos_oculta_saida.T)
    delta_oculta = erro_oculta * sigmoid_derivada(camada_oculta)

    # Atualização dos pesos e bias
    pesos_oculta_saida += camada_oculta.T.dot(delta_saida) * taxa_aprendizado
    pesos_input_oculta += treino_inputs.T.dot(delta_oculta) * taxa_aprendizado
    bias_saida += np.sum(delta_saida, axis=0, keepdims=True) * taxa_aprendizado
    bias_oculta += np.sum(delta_oculta, axis=0, keepdims=True) * taxa_aprendizado

    # Exibir erro a cada 1000 épocas
    if epoca % 1000 == 0:
        print(f"Época {epoca}, Erro: {np.mean(np.abs(erro_saida)):.4f}")

# Teste com os dados de treinamento
acertos = 0
for entrada, saida_esperada in zip(treino_inputs, treino_saidas):
    camada_oculta = sigmoid(np.dot(entrada, pesos_input_oculta) + bias_oculta)
    saida_obtida = sigmoid(np.dot(camada_oculta, pesos_oculta_saida) + bias_saida)
    saida_final = 1 if saida_obtida >= 0.5 else 0
    print(f"Entrada: {entrada}, Esperado: {saida_esperada[0]}, Obtido: {saida_final}")
    if saida_final == saida_esperada[0]:
        acertos += 1

taxa_acerto = (acertos / len(treino_inputs)) * 100
print(f"Taxa de acerto: {taxa_acerto:.2f}%")
