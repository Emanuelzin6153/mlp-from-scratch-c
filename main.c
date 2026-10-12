#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE 2
#define HIDDEN_SIZE 4
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.5f
#define EPOCHS 20000

// Função de ativação Sigmoid
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivada da Sigmoid (usada no backpropagation)
float d_sigmoid(float y) {
    return y * (1.0f - y);
}

// Inicializa pesos com valores aleatórios pequenos
float random_weight() {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

int main() {
    srand(time(NULL));

    // Dados de Treino do XOR
    float X[4][INPUT_SIZE] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };
    float Y[4][OUTPUT_SIZE] = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    // Pesos e Biases
    // W1: Camada Oculta, W2: Camada de Saída
    float W1[INPUT_SIZE][HIDDEN_SIZE];
    float B1[HIDDEN_SIZE];
    float W2[HIDDEN_SIZE][OUTPUT_SIZE];
    float B2[OUTPUT_SIZE];

    // Inicialização aleatória dos pesos
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            W1[i][j] = random_weight();
        }
    }
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        B1[j] = random_weight();
    }
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            W2[i][j] = random_weight();
        }
    }
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        B2[j] = random_weight();
    }

    printf("[INFO] Iniciando treino da MLP em C puro (%d épocas)...\n", EPOCHS);

    // Loop de Treino
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.0f;

        for (int p = 0; p < 4; p++) {
            // --- 1. FORWARD PASS ---
            // Camada Oculta
            float hidden_layer[HIDDEN_SIZE];
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                float sum = B1[j];
                for (int i = 0; i < INPUT_SIZE; i++) {
                    sum += X[p][i] * W1[i][j];
                }
                hidden_layer[j] = sigmoid(sum);
            }

            // Camada de Saída
            float output_layer[OUTPUT_SIZE];
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float sum = B2[j];
                for (int i = 0; i < HIDDEN_SIZE; i++) {
                    sum += hidden_layer[i] * W2[i][j];
                }
                output_layer[j] = sigmoid(sum);
            }

            // Cálculo do Erro (MSE)
            float error = Y[p][0] - output_layer[0];
            total_loss += error * error;

            // --- 2. BACKPROPAGATION ---
            // Gradiente da saída
            float d_output = error * d_sigmoid(output_layer[0]);

            // Gradiente da camada oculta
            float d_hidden[HIDDEN_SIZE];
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                d_hidden[i] = d_output * W2[i][0] * d_sigmoid(hidden_layer[i]);
            }

            // Atualização de Pesos e Biases (W2 e B2)
            for (int i = 0; i < HIDDEN_SIZE; i++) {
                W2[i][0] += LEARNING_RATE * d_output * hidden_layer[i];
            }
            B2[0] += LEARNING_RATE * d_output;

            // Atualização de Pesos e Biases (W1 e B1)
            for (int i = 0; i < INPUT_SIZE; i++) {
                for (int j = 0; j < HIDDEN_SIZE; j++) {
                    W1[i][j] += LEARNING_RATE * d_hidden[j] * X[p][i];
                }
            }
            for (int j = 0; j < HIDDEN_SIZE; j++) {
                B1[j] += LEARNING_RATE * d_hidden[j];
            }
        }

        // Mostrar progresso a cada 5000 épocas
        if (epoch % 5000 == 0) {
            printf("Época %5d | Erro Total (Loss): %.6f\n", epoch, total_loss / 4.0f);
        }
    }

    printf("\n[SUCESSO] Treino Concluído! Testando previsões finais:\n");
    printf("----------------------------------------------------\n");

    // Teste Final da Rede Treinada
    for (int p = 0; p < 4; p++) {
        float hidden_layer[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float sum = B1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += X[p][i] + 0.0f; // ajuste de sintaxe
                // Corrigindo para multiplicação correta de pesos:
            }
        }
    }

    // Loop limpo de inferência para exibição
    for (int p = 0; p < 4; p++) {
        float hidden_layer[HIDDEN_SIZE];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            float sum = B1[j];
            for (int i = 0; i < INPUT_SIZE; i++) {
                sum += X[p][i] * W1[i][j];
            }
            hidden_layer[j] = sigmoid(sum);
        }

        float output = 0.0f;
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            output += hidden_layer[i] * W2[i][0];
        }
        output = sigmoid(output + B2[0]);

        printf("Entrada: [%.0f, %.0f] | Previsto: %.4f (Arredondado: %.0f) | Real: %.0f\n",
               X[p][0], X[p][1], output, roundf(output), Y[p][0]);
    }

    return 0;
}
