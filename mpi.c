#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM

#define M 10
#define N 1000
#define MAX_ITERATIONS 10000
#define ALPHA 0.1
#define ACCURACY_TORLERANCE 0.0

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

    printf("size = %d\n", size);

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

    double **localInputs = (double **)malloc(sizeof(double *) * localN);
    for (int i = 0; i < localN; i++)
        localInputs[i] = (double *)malloc(sizeof(double) * M);
    double *localOutputs = (double *)malloc(sizeof(double) * localN);

    if (rank == 0)
    {
        for (int i = 0; i < localN; i++)
            localInputs[i] = inputs[i];

        for (int i = localN; i < N;)
        {
            int upper = i + localN;
            int dest = i / localN;
            for (; i < upper; i++)
            {
                MPI_Ssend(inputs[i], M, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            }
        }
    }
    else
    {
        for (int i = 0; i < localN; i++)
        {
            MPI_Status status;
            MPI_Recv(localInputs[i], M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        }
    }
    MPI_Scatter(outputs, localN, MPI_DOUBLE, localOutputs, localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // printf("Inputs outputs recieved by %d:\n", rank);
    // for (int _i = 0; _i < localN; _i++)
    // {
    //     printf("output - %lf : inputs - ", localOutputs[_i]);
    //     for (int _j = 0; _j < M; _j++)
    //         printf("%lf ", localInputs[_i][_j]);
    //     printf("\n");
    // }

    // for (int i = 0; i < localN; i++)
    // {
    //     if (localOutputs[i] != outputs[i])
    //         puts("outputs dont match");
    // }
    // puts("outputs checking over...");

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        // printf("Thetas sending by %d: ", rank);
        // for (int _i = 0; _i < M; _i++)
        //     printf("%lf ", theta[_i]);
        // printf("\n");

        MPI_Bcast(theta, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // printf("Thetas recieved by %d: ", rank);
        // for (int _i = 0; _i < M; _i++)
        //     printf("%lf ", theta[_i]);
        // printf("\n");

        double newTheta[M];

        for (int k = 0; k < M; k++)
        {
            double localT = 0;
            for (int n = 0; n < localN; n++)
            {
                double h = 0;
                for (int i = 0; i < M; i++)
                {
                    h += localInputs[n][i] * theta[i];
                }
                localT += (h - localOutputs[n]) * localInputs[n][k];
            }

            double t = 0;
            MPI_Reduce(&localT, &t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                t = theta[k] - ALPHA * localT / N;
                newTheta[k] = t;
            }
        }

        if (rank == 0)
        {
            for (int i = 0; i < M; i++)
                theta[i] = newTheta[i];
        }
    }

    if (rank != 0)
        for (int i = 0; i < M; i++)
            free(localInputs[i]);
    free(localInputs);
    free(localOutputs);

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