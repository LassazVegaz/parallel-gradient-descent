#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM

#define M 10
#define N 1000
#define MAX_ITERATIONS 1000
#define ALPHA 0.5
#define ACCURACY_TORLERANCE 0.0001

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

void checkThetaAccuracy(double *expectedTheta, double *theta)
{
    int thetasAreAccurate = 1;
    for (int i = 0; i < M; i++)
    {
        if (fabs(theta[i] - expectedTheta[i]) > ACCURACY_TORLERANCE)
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
        int inputRow = n * M;
        for (int i = 0; i < M; i++)
        {
            h += inputs[inputRow + i] * theta[i];
        }
        error += abs(h - outputs[n]);
    }
    error /= N;
    printf("error: %lf\n", error);
}

void printThetaMapping(double *expectedTheta, double *calculatedTheta)
{
    puts("Expected Thetas vs Computed Thetas");

    for (int i = 0; i < M; i++)
    {
        printf("%lf -> %lf\n", expectedTheta[i], calculatedTheta[i]);
    }
}

int main()
{
    int size, rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int localN = N / size;

    double *inputs;
    double *outputs;
    double *actualTheta;

    // theta are the coefficients we are trying to find
    double *theta = malloc(sizeof(double) * M);

    // init inputs, outputs and actual thetas in rank 0
    if (rank == 0)
    {
        inputs = malloc(sizeof(double) * M * N); // a one dimentional two dimentional like array
        outputs = malloc(sizeof(double) * N);
        actualTheta = malloc(sizeof(double) * M);
        init(inputs, outputs, actualTheta);

        for (int i = 0; i < M; i++)
            theta[i] = 0;
    }

    // dynamic arrays to store inputs and outputs in each rank
    double *localInputs = malloc(sizeof(double) * localN * M); // a 2D array like 1D array
    double *localOutputs = malloc(sizeof(double) * localN);

    // time should be counted from here because data spread is part of MPI
    // if it wasnt MPI, data doesnt need to be spread
    double startTime = MPI_Wtime();

    // distribute inputs
    MPI_Scatter(inputs, localN * M, MPI_DOUBLE, localInputs, localN * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // distribute outputs
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
                int inputRow = n * M;
                for (int i = 0; i < M; i++)
                {
                    h += localInputs[inputRow + i] * theta[i];
                }
                localT += (h - localOutputs[n]) * localInputs[inputRow + k];
            }

            // reduce all totals into one
            double t = 0;
            MPI_Reduce(&localT, &t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0)
            {
                t = theta[k] - ALPHA * t / N;
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

    free(localInputs);
    free(localOutputs);

    if (rank == 0)
    {
        printf("Time taken = %lf\n", endTime - startTime);

        // print mapping
        printThetaMapping(actualTheta, theta);

        // check if thetas are accurate
        checkThetaAccuracy(actualTheta, theta);

        // check error
        printError(inputs, outputs, theta);

        // clean up
        free(inputs);
        free(outputs);
        free(actualTheta);
        free(theta);
    }

    MPI_Finalize();

    return 0;
}

// Time taken = 0.232665