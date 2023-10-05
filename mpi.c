#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM

#define M 10
#define N 1000
#define MAX_ITERATIONS 1000
#define ALPHA 0.1
#define ACCURACY_TORLERANCE 0.001

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
        double h = 0;
        for (int i = 0; i < M; i++)
        {
            h += inputs[n][i] * theta[i];
        }
        error += abs(h - outputs[n]);
    }
    error /= N;
    printf("error: %lf\n", error);
}

int main()
{
    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int localN = N / size;

    double inputs[N][M];
    double outputs[N];
    double actualTheta[M];

    // theta are the coefficients we are trying to find
    double theta[M];

    if (rank == 0)
    {
        init(inputs, outputs, actualTheta);
        for (int i = 0; i < M; i++)
            theta[i] = 0;
    }

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        MPI_Bcast(theta, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        double newTheta[M];

        for (int k = 0; k < M; k++)
        {
            double *localInputs = (double *)malloc(sizeof(double) * localN);
            double *localOutputs = (double *)malloc(sizeof(double) * localN);
            MPI_Scatter(inputs, localN, MPI_DOUBLE, localInputs, localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Scatter(inputs, localN, MPI_DOUBLE, localOutputs, localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            double localT = 0;
            for (int n = 0; n < localN; n++)
            {
                double h = 0;
                for (int i = 0; i < M; i++)
                {
                    h += inputs[n][i] * theta[i];
                }
                localT += (h - outputs[n]) * inputs[n][k];
            }

            free(localInputs);
            free(localOutputs);

            double t = 0;
            MPI_Reduce(&localT, &t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                t = theta[k] - ALPHA * localT / N;
                newTheta[k] = localT;
            }
        }

        if (rank == 0)
        {
            for (int i = 0; i < M; i++)
                theta[i] = newTheta[i];
        }
    }

    if (rank == 0)
    {
        // check if thetas are accurate
        checkThetaAccuracy(theta, actualTheta);

        // check error
        printError(inputs, outputs, theta);
    }

    MPI_Finalize();

    return 0;
}