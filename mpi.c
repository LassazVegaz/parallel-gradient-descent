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

void init(double **inputs, double *outputs, double *theta)
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

void printError(double **inputs, double *outputs, double *theta)
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

    double **inputs;
    double *outputs;
    double *actualTheta;

    // theta are the coefficients we are trying to find
    double *theta = malloc(sizeof(double) * M);

    // init inputs, outputs and actual thetas in rank 0
    if (rank == 0)
    {
        inputs = malloc(sizeof(double *) * N);
        for (int i = 0; i < N; i++)
            inputs[i] = malloc(sizeof(double) * M);
        outputs = malloc(sizeof(double) * N);
        actualTheta = malloc(sizeof(double) * M);
        init(inputs, outputs, actualTheta);

        for (int i = 0; i < M; i++)
            theta[i] = 0;
    }

    // dynamic arrays to store inputs and outputs in each rank
    double **localInputs = (double **)malloc(sizeof(double *) * localN); // a multi array
    for (int i = 0; i < localN; i++)
        localInputs[i] = (double *)malloc(sizeof(double) * M);
    double *localOutputs = (double *)malloc(sizeof(double) * localN);

    // time should be counted from here because data spread if part of MPI
    // if it wasnt MPI, data doesnt need to be spread
    double startTime = MPI_Wtime();

    // send each input row to other ranks
    // a multidim array cannot be sent as a whole in MPI
    // but still this is efficient
    if (rank == 0)
    {
        for (int i = 0; i < N;)
        {
            int upper = i + localN;
            int dest = i / localN;
            for (; i < upper; i++)
            {
                int buffSize = sizeof(double) * M + MPI_BSEND_OVERHEAD;
                double *buff = malloc(buffSize);
                MPI_Buffer_attach(buff, buffSize);
                MPI_Bsend(inputs[i], M, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                MPI_Buffer_detach(buff, &buffSize);
                free(buff);
            }
        }
    }
    // gathering sent inputs
    for (int i = 0; i < localN; i++)
    {
        MPI_Status status;
        MPI_Recv(localInputs[i], M, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }

    // scatter outputs
    MPI_Scatter(outputs, localN, MPI_DOUBLE, localOutputs, localN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        // for iteration, thetas are updated. therefore distribute them
        MPI_Bcast(theta, M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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

            // reduce all totals into one
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

    double endTime = MPI_Wtime();

    for (int i = 0; i < localN; i++)
        free(localInputs[i]);
    free(localInputs);
    free(localOutputs);

    if (rank == 0)
    {
        printf("Time taken = %lf\n", endTime - startTime);

        // check if thetas are accurate
        checkThetaAccuracy(theta, actualTheta);

        // check error
        printError(inputs, outputs, theta);
    }

    if (rank == 0)
    {
        for (int i = 0; i < N; i++)
            free(inputs[i]);
        free(inputs);
        free(outputs);
        free(actualTheta);
        free(theta);
    }

    MPI_Finalize();

    return 0;
}