# Micro-MLP em C puro

Uma rede neural multicamadas (MLP) escrita do zero em C, sem frameworks e sem dependências além da biblioteca padrão. Ela aprende a função lógica **XOR**, o exemplo clássico de problema que um neurônio sozinho não resolve.

O objetivo é entender o que acontece por baixo de bibliotecas como PyTorch e TensorFlow: o forward pass, o cálculo do erro e o backpropagation, tudo escrito à mão.

## O que o projeto faz

- Treina uma rede 2 → 4 → 1 (2 entradas, 4 neurônios ocultos, 1 saída) para aprender a tabela do XOR:

  | Entrada | Saída esperada |
  |---------|----------------|
  | `[0, 0]` | `0` |
  | `[0, 1]` | `1` |
  | `[1, 0]` | `1` |
  | `[1, 1]` | `0` |

- **Forward pass:** somas ponderadas com ativação sigmoide, escritas com loops explícitos.
- **Backpropagation:** gradientes calculados manualmente e pesos atualizados por gradiente descendente estocástico (SGD), amostra por amostra.
- **Reinício automático:** o XOR às vezes trava num mínimo local, dependendo dos pesos iniciais. Se a loss final ficar acima de `0.01`, os pesos são sorteados de novo (até 5 tentativas).
- **Reprodutibilidade:** a seed usada é impressa no início de cada execução.

## Como compilar e executar

Qualquer compilador C compatível funciona (`gcc`, `clang`). A flag `-lm` liga a biblioteca matemática.

```bash
# Compilar
gcc -O2 -Wall -Wextra xor_mlp.c -o mlp_xor -lm

# Executar
./mlp_xor
```
**⚠️ Note on Compilation: Due to specific environment dependencies or local configuration quirks, this implementation runs out-of-the-box and seamlessly on online C compilers (like OnlineGDB or Replit), serving as a fast, portable sandbox to test the raw neural network logic.**

### Exemplo de saída

Os números variam a cada execução, porque os pesos iniciais dependem da seed.

```
[INFO] seed = 1792712315
[INFO] Tentativa 1/5
Época     0 | MSE: 0.270843
Época  5000 | MSE: 0.000565
Época 10000 | MSE: 0.000248
Época 15000 | MSE: 0.000157

[SUCESSO] MSE final: 0.000114
----------------------------------------------------
Entrada: [0, 0] | Previsto: 0.0079 (Arredondado: 0) | Real: 0
Entrada: [0, 1] | Previsto: 0.9888 (Arredondado: 1) | Real: 1
Entrada: [1, 0] | Previsto: 0.9891 (Arredondado: 1) | Real: 1
Entrada: [1, 1] | Previsto: 0.0123 (Arredondado: 0) | Real: 0
```

## Como funciona

**Arquitetura:** `Network` é uma `struct` com os pesos e biases das duas camadas (`W1`, `B1`, `W2`, `B2`). Todos os tamanhos são constantes definidas no topo do arquivo (`INPUT_SIZE`, `HIDDEN_SIZE`, `NUM_SAMPLES`).

**Forward:** cada neurônio calcula `sigmoid(bias + Σ entrada · peso)`, primeiro na camada oculta e depois na saída.

**Erro:** erro quadrático médio (MSE) sobre as 4 amostras.

**Backward:** com `error = alvo - saída`, os gradientes são:

```
d_saida   = error · σ'(saida)
d_oculta  = d_saida · W2 · σ'(oculta)
```

onde `σ'(y) = y · (1 - y)`, já que `y` é a saída da própria sigmoide. Depois, cada peso é atualizado com `peso += taxa_de_aprendizado · gradiente · entrada`.

**Hiperparâmetros** (no topo de `xor_mlp.c`): taxa de aprendizado `0.5`, `20000` épocas, `4` neurônios ocultos.

## Estrutura do repositório

```
.
├── xor_mlp.c   # implementação completa (rede, treino e teste)
├── README.md
├── LICENSE
└── .gitignore
```

## Limitações e próximos passos

Esta é uma implementação didática, com tamanhos fixos em tempo de compilação. Ideias para evoluir:

- [ ] Tamanhos de camada definidos em tempo de execução, com pesos alocados via `malloc`/`free`
- [ ] Função de multiplicação matriz-vetor reutilizável
- [ ] Número arbitrário de camadas ocultas
- [ ] Embaralhar a ordem das amostras a cada época
- [ ] Inicialização Xavier dos pesos
- [ ] Entropia cruzada no lugar de MSE (converge mais rápido com sigmoide)
- [ ] Testar em outros problemas além do XOR (AND, OR, pequenos datasets)

## Licença

Distribuído sob a licença [CC0 1.0](LICENSE) (domínio público).
