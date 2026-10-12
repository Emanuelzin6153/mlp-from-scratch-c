# Micro-MLP in Pure C 🧠⚙️

Uma implementação minimalista e de baixo nível de uma **Rede Neural Multicamadas (MLP)** escrita inteiramente em **C puro**, desenvolvida sem o uso de frameworks pesados (como PyTorch ou TensorFlow). 

O objetivo deste projeto é explorar a engenharia por trás do *Machine Learning*, lidando diretamente com alocação de memória, matrizes, propagação e retropropagação do erro (*Backpropagation*).

## 🚀 O que o projeto faz
- **Resolve o problema do XOR:** Treina uma rede neural para aprender a tabela lógica não-linear do XOR (`[0,0]->0`, `[0,1]->1`, `[1,0]->1`, `[1,1]->0`).
- **Forward Pass Manual:** Cálculo explícito de somas ponderadas e ativação por função **Sigmoid**.
- **Backpropagation Manual:** Propagação do erro, cálculo de gradientes e atualização de pesos e vieses (*biases*).
- **Zero Dependências:** Utiliza apenas bibliotecas padrão de C (`stdio.h`, `stdlib.h`, `math.h`, `time.h`).

## 🛠️ Tecnologias e Conceitos Envolvidos
- Linguagem C
- Álgebra Linear Aplicada (Multiplicação de Matrizes)
- Otimização e Gestão de Memória
- Sistemas de Inferência de Baixo Nível

## 📦 Como Compilar e Executar

Podes compilar o código utilizando qualquer compilador C compatível (como `gcc` ou `clang`), ativando a biblioteca matemática (`-lm`):

```bash
# Compilar
gcc main.c -o mlp_xor -lm

# Executar
./mlp_xor
