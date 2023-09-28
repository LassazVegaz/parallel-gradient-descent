// this is the serial implementation of Gradient Descent algorithm
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// f(x1, x2, x3, x4) = 3 * x1 - 7 * x2 + 5 * x3 - 4 * x4

#define M 4
#define N 1000
#define MAX_ITERATIONS 10000
#define ALPHA 0.1

/// @brief The function we are trying to find coefficients for
double f(double x1, double x2, double x3, double x4)
{
    return 3 * x1 - 7 * x2 + 5 * x3 - 4 * x4;
}

void init(double **inputs, double *outputs)
{
    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < M; k++)
        {
            inputs[i][k] = (double)rand() / RAND_MAX;
        }
        outputs[i] = f(inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]);
    }
}

int main()
{
    double inputs[N][M];
    double outputs[N];
    init(inputs, outputs);

    // theta are the coefficients we are trying to find
    float theta[M];
    for (int i = 0; i < M; i++)
        theta[i] = 0;

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        float newTheta[M];
        for (int i = 0; i < M; i++)
            newTheta[i] = 0;

        for (int k = 0; k < M; k++)
        {
            float t = 0;
            for (int n = 0; n < N; n++)
            {
                float h = 0;
                for (int i = 0; i < M; i++)
                {
                    h += inputs[n][i] * theta[i];
                }
                t += (h - outputs[n]) * inputs[n][k];
            }
            t = theta[k] - ALPHA * t / N;
            newTheta[k] = t;
        }

        for (int i = 0; i < M; i++)
            theta[i] = newTheta[i];
    }

    printf("theta: ");
    for (int i = 0; i < M; i++)
        printf("%f ", theta[i]);
    printf("\n");

    return 0;
}