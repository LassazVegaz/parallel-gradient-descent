// f(x1, x2, x3, ..., xM) = theta0 * x0 + theta1 * x1 + theta2 * x2 + ... + thetaM * xM
#include <CL/sycl.hpp>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <cstdlib>

#define M 10
#define N 1000
#define MAX_ITERATIONS 1000
#define ALPHA 0.1
#define ACCURACY_TORLERANCE 0.001
#define MAX_DECIMALS 4

using namespace sycl;

/// @brief The function we are trying to find coefficients for
float f(float *x, float *theta)
{
    float result = 0;
    for (int i = 0; i < M; i++)
    {
        result += theta[i] * x[i];
    }
    return result;
}

void init(float inputs[N][M], float outputs[N], float theta[M])
{
    srand(time(NULL));

    for (int i = 0; i < M; i++)
        theta[i] = (float)rand() / (float)RAND_MAX;

    for (int i = 0; i < N; i++)
    {
        for (int k = 0; k < M; k++)
        {
            // i th data point, k th variable
            inputs[i][k] = (float)rand() / (float)RAND_MAX;
        }
        outputs[i] = f(inputs[i], theta);
    }
}

void checkThetaAccuracy(float *theta, float *actualTheta)
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
        std::cout << "Thetas are accurate" << std::endl;
    else
        std::cout << "Thetas are not accurate" << std::endl;
}

void printError(float inputs[N][M], float outputs[N], float *theta)
{
    float error = 0;

    for (int n = 0; n < N; n++)
    {
        float h = f(inputs[n], theta);
        error += abs(h - outputs[n]);
    }

    error /= N;
    std::cout << std::fixed << std::setprecision(MAX_DECIMALS) << "error: " << error << std::endl;
}

void printThetaMapping(float *expectedTheta, float *calculatedTheta)
{
    std::cout << "Expected Thetas vs Computed Thetas" << std::endl;

    for (int i = 0; i < M; i++)
    {
        std::cout << expectedTheta[i] << "  ->  " << calculatedTheta[i] << std::endl;
    }
}

int main()
{
    queue q(gpu_selector_v);
    std::cout << "Device: " << q.get_device().get_info<info::device::name>() << std::endl;

    float inputs[N][M];
    float outputs[N];
    float actualTheta[M];
    init(inputs, outputs, actualTheta);

    // theta are the coefficients we are trying to find
    float theta[M];
    for (int i = 0; i < M; i++)
        theta[i] = 0;

    {
        buffer buf_inputs(*inputs, range(N, M));
        buffer buf_outputs(outputs, range(N));

        for (int i = 0; i < MAX_ITERATIONS; i++)
        {
            float newTheta[M];
            {
                buffer buf_theta(theta, range(M));
                buffer buf_newTheta(newTheta, range(M));

                q.submit([&](handler &h)
                         {
                accessor a_inputs(buf_inputs, h, read_only);
                accessor a_outputs(buf_outputs, h, read_only);
                accessor a_theta(buf_theta, h, read_only);
                accessor a_newTheta(buf_newTheta, h, write_only);

                h.parallel_for(range(M), [=](id<1> k) {
                    float t = 0;
                    for (int n = 0; n < N; n++)
                    {
                        float h = 0;
                        for (int i = 0; i < M; i++)
                        {
                            h += a_inputs[n][i] * a_theta[i];
                        }
                        t += (h - a_outputs[n]) * a_inputs[n][k];
                    }
                    t = a_theta[k] - ALPHA * t / N;
                    a_newTheta[k] = t;
                }); });
            }

            for (int i = 0; i < M; i++)
                theta[i] = newTheta[i];
        }
    }

    // check mapping
    printThetaMapping(actualTheta, theta);

    // check if thetas are accurate
    checkThetaAccuracy(theta, actualTheta);

    // check error
    printError(inputs, outputs, theta);

    return 0;
}