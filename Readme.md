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
   - As sinapses (pesos) são ajustadas com base no erro da previsão.
   - O treinamento ocorre por um número fixo de épocas ou até que não haja mais erros.

3. **Teste e Validação:**
   - Após o treinamento, o modelo avalia a taxa de acerto sobre os dados de entrada.
   - O usuário pode inserir novos padrões para prever sua classificação.

4. **Armazenamento:**
   - Os pesos finais são salvos em um arquivo `sinapses_perceptron.txt` para futuras previsões sem a necessidade de retreinar o modelo.


## 📌 Exemplo de Uso

Durante a execução, o usuário pode inserir novos padrões manualmente:
```bash
Digite 16 valores separados por espaço para representar a matriz 4x4: 
0 0 1 0 0 1 1 0 0 1 1 0 0 0 0 0
A rede neural prevê que este número é um: 1
```


