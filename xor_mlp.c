// MLP para XOR em C puro, v2
//  - tamanho da camada oculta definido em tempo de execução (malloc/free)
//  - multiplicação matriz-vetor reutilizável (matvec)
//
// Compilar: gcc -O2 -Wall -Wextra xor_mlp.c -o xor_mlp -lm
// Usar:     ./xor_mlp [neuronios_ocultos]    (padrão: 4)
// Depurar:  gcc -g -fsanitize=address,undefined xor_mlp.c -o xor_mlp_dbg -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define INPUT_SIZE     2
#define OUTPUT_SIZE    1
#define DEFAULT_HIDDEN 4
#define MAX_HIDDEN     1024
#define NUM_SAMPLES    4
#define LEARNING_RATE  0.5f
#define EPOCHS         20000
#define LOG_EVERY      5000
#define MAX_ATTEMPTS   5
#define LOSS_TARGET    0.01f

// Uma camada densa: n_out neurônios, cada um com n_in pesos.
typedef struct {
    int    n_in;
    int    n_out;
    float *W;      // n_out x n_in, uma linha por neurônio: W[j * n_in + i]
    float *b;      // n_out
    float *out;    // n_out, ativação (guardada para o backprop)
    float *delta;  // n_out, gradiente local
} Layer;

typedef struct {
    Layer hidden;
    Layer output;
} Network;

static const float X[NUM_SAMPLES][INPUT_SIZE] = {
    {0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}, {1.0f, 1.0f}
};
static const float Y[NUM_SAMPLES][OUTPUT_SIZE] = {{0.0f}, {1.0f}, {1.0f}, {0.0f}};

static float sigmoid(float x)   { return 1.0f / (1.0f + expf(-x)); }
static float d_sigmoid(float y) { return y * (1.0f - y); } // y já é sigmoid(x)

static float random_weight(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

// ---------- Gerenciamento de memória ----------

static void layer_free(Layer *l) {
    free(l->W);
    free(l->b);
    free(l->out);
    free(l->delta);
    l->W = l->b = l->out = l->delta = NULL;
}

// Devolve 0 em sucesso, -1 se faltar memória (nada fica vazando).
static int layer_create(Layer *l, int n_in, int n_out) {
    l->n_in  = n_in;
    l->n_out = n_out;
    l->W     = malloc((size_t)n_in * (size_t)n_out * sizeof *l->W);
    l->b     = malloc((size_t)n_out * sizeof *l->b);
    l->out   = malloc((size_t)n_out * sizeof *l->out);
    l->delta = malloc((size_t)n_out * sizeof *l->delta);
    if (!l->W || !l->b || !l->out || !l->delta) {
        layer_free(l);  // free(NULL) é seguro
        return -1;
    }
    return 0;
}

static int network_create(Network *net, int hidden_size) {
    if (layer_create(&net->hidden, INPUT_SIZE, hidden_size) != 0)
        return -1;
    if (layer_create(&net->output, hidden_size, OUTPUT_SIZE) != 0) {
        layer_free(&net->hidden);
        return -1;
    }
    return 0;
}

static void network_free(Network *net) {
    layer_free(&net->hidden);
    layer_free(&net->output);
}

static void layer_init_weights(Layer *l) {
    for (int k = 0; k < l->n_out * l->n_in; k++)
        l->W[k] = random_weight();
    for (int j = 0; j < l->n_out; j++)
        l->b[j] = random_weight();
}

static void network_init_weights(Network *net) {
    layer_init_weights(&net->hidden);
    layer_init_weights(&net->output);
}

// ---------- Operações de camada ----------

// y = W * x + b, com W (rows x cols) em ordem de linha
static void matvec(const float *W, const float *x, const float *b,
                   float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = &W[(size_t)r * (size_t)cols];
        float sum = b[r];
        for (int c = 0; c < cols; c++)
            sum += row[c] * x[c];
        y[r] = sum;
    }
}

static void layer_forward(Layer *l, const float *in) {
    matvec(l->W, in, l->b, l->out, l->n_out, l->n_in);
    for (int j = 0; j < l->n_out; j++)
        l->out[j] = sigmoid(l->out[j]);
}

// Passo de gradiente: usa l->delta já calculado e a entrada 'in' da camada
static void layer_update(Layer *l, const float *in, float lr) {
    for (int j = 0; j < l->n_out; j++) {
        float *row = &l->W[(size_t)j * (size_t)l->n_in];
        for (int i = 0; i < l->n_in; i++)
            row[i] += lr * l->delta[j] * in[i];
        l->b[j] += lr * l->delta[j];
    }
}

// ---------- Rede ----------

// Forward pass: devolve ponteiro para a saída (vive dentro da rede)
static const float *forward(Network *net, const float *x) {
    layer_forward(&net->hidden, x);
    layer_forward(&net->output, net->hidden.out);
    return net->output.out;
}

// Um passo de treino (SGD) numa amostra; devolve o erro quadrático
static float train_step(Network *net, const float *x, const float *target) {
    const float *out = forward(net, x);

    float sq_err = 0.0f;
    for (int k = 0; k < net->output.n_out; k++) {
        float err = target[k] - out[k];
        sq_err += err * err;
        net->output.delta[k] = err * d_sigmoid(out[k]);
    }

    // delta da camada oculta calculado ANTES de atualizar os pesos de saída
    for (int j = 0; j < net->hidden.n_out; j++) {
        float sum = 0.0f;
        for (int k = 0; k < net->output.n_out; k++)
            sum += net->output.delta[k] * net->output.W[(size_t)k * (size_t)net->output.n_in + j];
        net->hidden.delta[j] = sum * d_sigmoid(net->hidden.out[j]);
    }

    layer_update(&net->output, net->hidden.out, LEARNING_RATE);
    layer_update(&net->hidden, x, LEARNING_RATE);

    return sq_err;
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

int main(int argc, char **argv) {
    int hidden_size = DEFAULT_HIDDEN;
    if (argc > 1) {
        char *end;
        long v = strtol(argv[1], &end, 10);
        if (*end != '\0' || v < 1 || v > MAX_HIDDEN) {
            fprintf(stderr, "Uso: %s [neuronios_ocultos (1 a %d)]\n", argv[0], MAX_HIDDEN);
            return 1;
        }
        hidden_size = (int)v;
    }

    unsigned seed = (unsigned)time(NULL);
    srand(seed);
    printf("[INFO] seed = %u | neurônios ocultos = %d\n", seed, hidden_size);

    Network net;
    if (network_create(&net, hidden_size) != 0) {
        fprintf(stderr, "[ERRO] sem memória para criar a rede\n");
        return 1;
    }

    float mse = 1.0f;

    // XOR às vezes cai num mínimo local: se acontecer, reinicia os pesos
    for (int attempt = 1; attempt <= MAX_ATTEMPTS && mse > LOSS_TARGET; attempt++) {
        printf("[INFO] Tentativa %d/%d\n", attempt, MAX_ATTEMPTS);
        network_init_weights(&net);
        mse = train(&net);
    }

    printf("\n[%s] MSE final: %.6f\n", mse <= LOSS_TARGET ? "SUCESSO" : "AVISO", mse);
    printf("----------------------------------------------------\n");

    for (int p = 0; p < NUM_SAMPLES; p++) {
        const float *out = forward(&net, X[p]);
        printf("Entrada: [%.0f, %.0f] | Previsto: %.4f (Arredondado: %.0f) | Real: %.0f\n",
               X[p][0], X[p][1], out[0], roundf(out[0]), Y[p][0]);
    }

    network_free(&net);
    return 0;
}
