#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM

#define M 10
#define N 1000
#define MAX_ITERATIONS 1000
#define ALPHA 0.1
#define ACCURACY_TORLERANCE 0.00

/// @brief The function we are trying to find coefficients for
double f(double *x, double *theta)
{
    double result = 0;
    for (int i = 0; i < M; i++)
    {
        result += theta[i] * x[i];
    }
    return result;
}

void init(double inputs[N][M], double outputs[N], double theta[M])
{
    srand(time(NULL));

    for (int i = 0; i < M; i++)
        theta[i] = (double)rand() / RAND_MAX;

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < M; k++)
        {
            // i th data point, k th variable
            inputs[i][k] = (double)rand() / RAND_MAX;
        }
        outputs[i] = f(inputs[i], theta);
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

void printError(double inputs[N][M], double outputs[N], double *theta)
{
    double error = 0;

    for (int n = 0; n < N; n++)
    {
        double h = f(inputs[n], theta);
        error += abs(h - outputs[n]);
    }

    error /= N;
    printf("error: %lf\n", error);
}

int main()
{
    double inputs[N][M];
    double outputs[N];
    double actualTheta[M];
    init(inputs, outputs, actualTheta);

    // theta are the coefficients we are trying to find
    double theta[M];
    for (int i = 0; i < M; i++)
        theta[i] = 0;

    double tstart = omp_get_wtime();

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        double newTheta[M];

        for (int k = 0; k < M; k++)
        {
            double t = 0;
            for (int n = 0; n < N; n++)
            {
                double h = 0;
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

    double tend = omp_get_wtime();
    double ttime = tend - tstart;
    printf("Time taken = %lf\n", ttime);

    // check if thetas are accurate
    checkThetaAccuracy(theta, actualTheta);

    // check error
    printError(inputs, outputs, theta);

    return 0;
}