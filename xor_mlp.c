// MLP para XOR em C puro: 2 entradas -> 4 neurônios ocultos -> 1 saída
// Compilar: gcc -O2 -Wall -Wextra xor_mlp.c -o xor_mlp -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE    2
#define HIDDEN_SIZE   4
#define NUM_SAMPLES   4
#define LEARNING_RATE 0.5f
#define EPOCHS        20000
#define LOG_EVERY     5000
#define MAX_ATTEMPTS  5
#define LOSS_TARGET   0.01f

typedef struct {
    float W1[INPUT_SIZE][HIDDEN_SIZE];
    float B1[HIDDEN_SIZE];
    float W2[HIDDEN_SIZE];
    float B2;
} Network;

static const float X[NUM_SAMPLES][INPUT_SIZE] = {
    {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
};
static const float Y[NUM_SAMPLES] = {0.0f, 1.0f, 1.0f, 0.0f};

static float sigmoid(float x)   { return 1.0f / (1.0f + expf(-x)); }
static float d_sigmoid(float y) { return y * (1.0f - y); } // y já é sigmoid(x)

static float random_weight(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

static void init_network(Network *net) {
    for (int i = 0; i < INPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W1[i][j] = random_weight();
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        net->B1[j] = random_weight();
        net->W2[j] = random_weight();
    }
    net->B2 = random_weight();
}

// Forward pass: devolve a saída e preenche 'hidden' (necessário no backprop)
static float forward(const Network *net, const float *x, float *hidden) {
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        float sum = net->B1[j];
        for (int i = 0; i < INPUT_SIZE; i++)
            sum += x[i] * net->W1[i][j];
        hidden[j] = sigmoid(sum);
    }
    float sum = net->B2;
    for (int j = 0; j < HIDDEN_SIZE; j++)
        sum += hidden[j] * net->W2[j];
    return sigmoid(sum);
}

// Um passo de treino (SGD) numa amostra; devolve o erro quadrático
static float train_step(Network *net, const float *x, float target) {
    float hidden[HIDDEN_SIZE];
    float out   = forward(net, x, hidden);
    float error = target - out;

    float d_out = error * d_sigmoid(out);
    float d_hidden[HIDDEN_SIZE];
    for (int j = 0; j < HIDDEN_SIZE; j++)  // calculado ANTES de atualizar W2
        d_hidden[j] = d_out * net->W2[j] * d_sigmoid(hidden[j]);

    for (int j = 0; j < HIDDEN_SIZE; j++) {
        net->W2[j] += LEARNING_RATE * d_out * hidden[j];
        net->B1[j] += LEARNING_RATE * d_hidden[j];
        for (int i = 0; i < INPUT_SIZE; i++)
            net->W1[i][j] += LEARNING_RATE * d_hidden[j] * x[i];
    }
    net->B2 += LEARNING_RATE * d_out;

    return error * error;
}

// Treina e devolve a loss (MSE) da última época
static float train(Network *net) {
    float mse = 0.0f;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total = 0.0f;
        for (int p = 0; p < NUM_SAMPLES; p++)
            total += train_step(net, X[p], Y[p]);
        mse = total / NUM_SAMPLES;

        if (epoch % LOG_EVERY == 0)
            printf("Época %5d | MSE: %.6f\n", epoch, mse);
    }
    return mse;
}

int main(void) {
    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    printf("[INFO] seed = %u\n", seed);

    Network net;
    float mse = 1.0f;

    // XOR às vezes cai num mínimo local: se acontecer, reinicia os pesos
    for (int attempt = 1; attempt <= MAX_ATTEMPTS && mse > LOSS_TARGET; attempt++) {
        printf("[INFO] Tentativa %d/%d\n", attempt, MAX_ATTEMPTS);
        init_network(&net);
        mse = train(&net);
    }

    printf("\n[%s] MSE final: %.6f\n", mse <= LOSS_TARGET ? "SUCESSO" : "AVISO", mse);
    printf("----------------------------------------------------\n");

    for (int p = 0; p < NUM_SAMPLES; p++) {
        float hidden[HIDDEN_SIZE];
        float out = forward(&net, X[p], hidden);
        printf("Entrada: [%.0f, %.0f] | Previsto: %.4f (Arredondado: %.0f) | Real: %.0f\n",
               X[p][0], X[p][1], out, roundf(out), Y[p]);
    }
    return 0;
}
