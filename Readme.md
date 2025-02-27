# Perceptron para Reconhecimento de Dígitos em Matriz 4x4

Este projeto implementa um Perceptron simples para reconhecer padrões numéricos representados por uma matriz 4x4 binária.

## 📌 Descrição
O código treina um Perceptron para classificar números representados em uma matriz 4x4, onde os valores 1 e 0 indicam a presença ou ausência de um traço do número.
Após o treinamento, o modelo pode prever novas entradas e salvar os pesos treinados em um arquivo para uso posterior.


## 🔧 Como Funciona

1. **Inicialização:**
   - Os pesos (sinapses) são iniciados aleatoriamente.
   - Um viés (bias) é definido.

2. **Treinamento:**
   - Para cada entrada, o Perceptron calcula a soma ponderada dos valores e aplica uma função de ativação.
   - Os pesos são ajustados com base no erro da previsão.
   - O treinamento ocorre por um número fixo de épocas ou até que não haja mais erros.

3. **Teste e Validação:**
   - Após o treinamento, o modelo avalia a taxa de acerto sobre os dados de entrada.
   - O usuário pode inserir novos padrões para prever sua classificação.

4. **Armazenamento:**
   - Os pesos finais são salvos em um arquivo `sinapses_perceptron.txt` para futuras previsões sem a necessidade de retreinar o modelo.

## 🖥️ Explicação do Código

- **Importação de Bibliotecas**
  ```python
  import numpy as np
  ```
  - Utiliza `numpy` para operações matemáticas e manipulação de arrays.

- **Definição dos Dados de Treinamento**
  ```python
  treino_inputs = np.array([...])
  treino_saidas = np.array([...])
  ```
  - Representa os números em formato binário, onde cada matriz 4x4 é convertida em um vetor de 16 elementos.
  - `treino_saidas` contém os rótulos correspondentes.

- **Inicialização das Sinapses**
  ```python
  sinapses = np.random.randint(0, 2, size=16).astype(float)
  bias = 1
  taxa_aprendizado = 0.5
  ```
  - Os pesos das conexões são iniciados aleatoriamente entre 0 e 1.
  - Define-se um viés para ajustar a ativação do neurônio.
  - A taxa de aprendizado determina o impacto da correção dos erros.

- **Função de Ativação**
  ```python
  def func_ativacao(soma):
      return 1 if soma >= 0 else 0
  ```
  - Utiliza uma função de ativação degrau para definir a saída do Perceptron.

- **Treinamento do Perceptron**
  ```python
  for epoca in range(100):
      erro_total = 0
      for entrada, saida_esperada in zip(treino_inputs, treino_saidas):
          soma = np.dot(entrada, sinapses) - bias
          saida_obtida = func_ativacao(soma)
          erro = saida_esperada - saida_obtida
          erro_total += abs(erro)
          sinapses += taxa_aprendizado * erro * entrada
      if erro_total == 0:
          break
  ```
  - Executa o aprendizado por 100 épocas ou até o erro ser 0.
  - Ajusta os pesos com base no erro entre a saída esperada e a obtida.

- **Teste do Modelo**
  ```python
  for entrada, saida_esperada in zip(treino_inputs, treino_saidas):
      soma = np.dot(entrada, sinapses) - bias
      saida_obtida = func_ativacao(soma)
      print(f"Entrada: {entrada}, Esperado: {saida_esperada}, Obtido: {saida_obtida}")
  ```
  - Avalia a taxa de acerto do modelo.

- **Previsão de Novos Valores**
  ```python
  def prever_usuario():
      sinapses, bias = carregar_sinapses()
      entrada = np.array([int(x) for x in input("Digite 16 valores: ").split()])
      soma = np.dot(entrada, sinapses) - bias
      print(f"A rede neural prevê que este número é um: {func_ativacao(soma)}")
  ```
  - Permite que o usuário insira novos padrões para prever sua classificação.



## 📌 Exemplo de Uso

Durante a execução, o usuário pode inserir novos padrões manualmente:
```bash
Digite 16 valores separados por espaço para representar a matriz 4x4: 
0 0 1 0 0 1 1 0 0 1 1 0 0 0 0 0
A rede neural prevê que este número é um: 1
```



