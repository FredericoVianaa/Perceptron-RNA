import numpy as np

# Definição dos dados de treinamento
treino_inputs = np.array([
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Número 1
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # Número 1
    [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Número 1
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],  # Número 1
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0],  # Número 0
    [1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0],  # Número 0
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],  # Número 0
    [1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1]   # Número 0
])

treino_saidas = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # Rótulos dos números

# Inicialização das sinapses iniciais e configs básicas
sinapses = np.random.randint(0, 2, size=16).astype(float)
bias = 1
taxa_aprendizado = 0.5

def func_ativacao(soma):
    return 1 if soma >= 0 else 0

# Treinamento do Perceptron
for epoca in range(100):  # Número de épocas
    erro_total = 0
    for entrada, saida_esperada in zip(treino_inputs, treino_saidas):
        soma = np.dot(entrada, sinapses) - bias
        saida_obtida = func_ativacao(soma)
        erro = saida_esperada - saida_obtida
        erro_total += abs(erro)
        sinapses += taxa_aprendizado * erro * entrada
    if erro_total == 0:
        break

# Salvando as sinapses em um arquivo txt
with open("sinapses_perceptron.txt", "w") as f:
    f.write(" ".join(map(str, sinapses)) + "\n")
    f.write(str(bias) + "\n")

print("Treinamento concluído. Sinapses salvas.")

# Testando com os exemplos treinados
acertos = 0
for entrada, saida_esperada in zip(treino_inputs, treino_saidas):
    soma = np.dot(entrada, sinapses) - bias
    saida_obtida = func_ativacao(soma)
    print(f"Entrada: {entrada}, Esperado: {saida_esperada}, Obtido: {saida_obtida}")
    if saida_obtida == saida_esperada:
        acertos += 1

taxa_acerto = (acertos / len(treino_inputs)) * 100
print(f"Taxa de acerto: {taxa_acerto:.2f}%")

# Função para carregar as sinapses de um arquivo txt
def carregar_sinapses():
    with open("sinapses_perceptron.txt", "r") as f:
        linhas = f.readlines()
        sinapses = np.array([float(x) for x in linhas[0].split()])
        bias = float(linhas[1])
    return sinapses, bias

# Função para prever um novo conjunto
def prever_usuario():
    sinapses, bias = carregar_sinapses()
    
    while True:
        entrada = input("Digite 16 valores separados por espaço para representar a matriz 4x4: ")
        entrada = np.array([int(x) for x in entrada.split()])
        
        if len(entrada) != 16:
            print("Erro: É necessário inserir exatamente 16 valores.")
            continue
        
        soma = np.dot(entrada, sinapses) - bias
        saida_obtida = func_ativacao(soma)
        print(f"A rede neural prevê que este número é um: {saida_obtida}")
        
        testar_novo = input("Deseja testar outro conjunto? (s/n): ")
        if testar_novo.lower() != 's':
            break

# Chamando a função para prever um novo conjunto
testar_novo = input("Deseja testar um novo conjunto? (s/n): ")
if testar_novo.lower() == 's':
    prever_usuario()