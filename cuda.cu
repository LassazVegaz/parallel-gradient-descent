#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM

#define M 10
#define N 1000
#define MAX_ITERATIONS 1000
#define ALPHA 0.1
#define ACCURACY_TORLERANCE 0.001

// error handling helper
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/// @brief The function we are trying to find coefficients for
double f(double *x, int x_row, double *theta)
{
    int row = x_row * M;
    double result = 0;

    for (int i = 0; i < M; i++)
    {
        result += theta[i] * x[row + i];
    }

    return result;
}

void init(double inputs[N * M], double outputs[N], double theta[M])
{
    srand(time(NULL));

    for (int i = 0; i < M; i++)
        theta[i] = (double)rand() / RAND_MAX;

    for (int i = 0; i < N; i++)
    {
        int upper = (i + 1) * M;
        for (int k = i * M; k < upper; k++)
        {
            inputs[k] = (double)rand() / RAND_MAX;
        }
        outputs[i] = f(inputs, i, theta);
    }
}

void checkThetaAccuracy(double *theta, double *actualTheta)
{
    int thetasAreAccurate = 1;
    for (int i = 0; i < M; i++)
    {
        if (abs(theta[i] - actualTheta[i]) > ACCURACY_TORLERANCE)
        {
            thetasAreAccurate = 0;
            break;
        }
    }
    if (thetasAreAccurate)
        printf("Thetas are accurate\n");
    else
        printf("Thetas are not accurate\n");
}

void printError(double inputs[N * M], double outputs[N], double *theta)
{
    double error = 0;
    for (int n = 0; n < N; n++)
    {
        double h = 0;
        for (int i = 0; i < M; i++)
        {
            h += inputs[n * M + i] * theta[i];
        }
        error += abs(h - outputs[n]);
    }
    error /= N;
    printf("error: %lf\n", error);
}

void printThetaMapping(double *expectedTheta, double *calculatedTheta)
{

    puts("Expected theta vs computed theta");

    for (int i = 0; i < M; i++)
    {
        printf("%lf  ->  %lf\n", expectedTheta[i], calculatedTheta[i]);
    }
}

__global__ void ComputeThetas(double *inputs, double *outputs, double *theta, double *newTheta)
{
    int k = threadIdx.x;
    double t = 0;

    for (int n = 0; n < N; n++)
    {
        int input_row = n * M;
        double h = 0;

        for (int i = 0; i < M; i++)
        {
            h += inputs[input_row + i] * theta[i];
        }

        t += (h - outputs[n]) * inputs[input_row + k];
    }

    t = theta[k] - ALPHA * t / N;
    newTheta[k] = t;
}

int main()
{
    double inputs[N * M]; // a 1D array instead of 2D because it is easy to copy in oneAPI
    double outputs[N];
    double actualTheta[M];
    init(inputs, outputs, actualTheta);

    // theta are the coefficients we are trying to find
    double theta[M];
    for (int i = 0; i < M; i++)
        theta[i] = 0;

    const int inputs_size = N * M * sizeof(double);
    const int n_size = N * sizeof(double);
    const int m_size = M * sizeof(double);

    double *d_inputs;
    double *d_outputs;
    double *d_theta;
    double *d_newTheta;
    cudaMalloc(&d_inputs, inputs_size);
    cudaMalloc(&d_outputs, n_size);
    cudaMalloc(&d_theta, m_size);
    cudaMalloc(&d_newTheta, m_size);

    cudaMemcpy(d_inputs, inputs, inputs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_outputs, outputs, n_size, cudaMemcpyHostToDevice);

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        cudaMemcpy(d_theta, theta, m_size, cudaMemcpyHostToDevice);

        ComputeThetas<<<1, M>>>(d_inputs, d_outputs, d_theta, d_newTheta);

        double *newTheta = (double *)malloc(m_size);
        cudaMemcpy(newTheta, d_newTheta, m_size, cudaMemcpyDeviceToHost);

        for (int i = 0; i < M; i++)
            theta[i] = newTheta[i];

        free(newTheta);
    }

    cudaFree(d_inputs);
    cudaFree(d_outputs);
    cudaFree(d_theta);
    cudaFree(d_newTheta);

    // print theta mappins
    printThetaMapping(actualTheta, theta);

    // check if thetas are accurate
    checkThetaAccuracy(theta, actualTheta);

    // check error
    printError(inputs, outputs, theta);

    return 0;
}
