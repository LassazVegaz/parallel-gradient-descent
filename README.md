# Parallel implementation of Gradient Descent

This is the parallel version of the gradient descent algorithm, implemented in C and C++. It is expected to use OpenMP, MPI, oneAPI and CUDA to implement the algorithm.
[serial.c](./serial.c) includes the serial version of the algorithm.

## Gradient Descent

The gradient descent algorithm is an iterative optimization algorithm used to minimize a function by 
iteratively moving in the direction of steepest descent as defined by the negative of the gradient. 
The algorithm is commonly used in machine learning and artificial intelligence to optimize models and 
minimize loss functions.

The gradient descent algorithm is defined by the following equation:

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta%20%3A%3D%20%5Ctheta%20-%20%5Calpha%20%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D)

where ![equation](https://latex.codecogs.com/gif.latex?%5Calpha) is the learning rate and

![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20J%28%5Ctheta%29%7D%7B%5Cpartial%20%5Ctheta%7D) is the gradient of the loss function.

## Parallelism

This implementation of the gradient descent algorithm is parallelized using multiple programming models 
to take advantage of the parallel processing capabilities of modern CPUs and GPUs. The OpenMP and MPI 
versions are designed for shared-memory and distributed-memory systems, respectively. The oneAPI version 
is designed to take advantage of the heterogeneous computing capabilities of modern CPUs and GPUs. The 
CUDA version is designed specifically for NVIDIA GPUs.

This project is intended for educational purposes and to demonstrate the performance benefits of parallel 
programming. Enjoy!

## Usage

### Environment

I used an Intel DevCloud environment to test the OpenMP version of the program. Search for Intel DevCloud and create an account. The Intel DevCloud has a Linux environment with the Intel C/C++ compiler and other nessesary tools for parallel programming.

If you are using VS Code, there is a way to code in the Intel DevCloud using VS Code. They have instructions for that. If you follow their instructions you will find a way to easily code in the DevCloud environment using VS Code.

I also have created some VS Code tasks to compile and run some C/C++ files for VS Code users. Tasks name are self-explanatory. Check them out.

### Serial

[serial.c](./serial.c) contains the serial version of the program. This program can be compiled and run on any system with a C compiler. To compile the serial version of the program, run the following command:

```bash
gcc -o serial serial.c
```

To run the program, run the following command:

```bash
./serial
```

### OpenMP

[omp.c](./omp.c) contains the OpenMP version of the program. Compile the code running the following command:

```bash
gcc -fopenmp -o omp omp.c
```

To run the program, run the following command:

```bash
qsub omp.pbs
```

### MPI

[mpi.c](./mpi.c) contains the MPI version of the program. Comiple it by running:

```bash
mpicc -o mpi mpi.c
```

To run the program, run the following command:

```bash
qsub mpi.pbs
```

### oneAPI

not yet implemented

### CUDA

not yet implemented

## References

[1] [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)

[2] [OpenMP](https://www.openmp.org/)

[3] [MPI](https://www.mpi-forum.org/)

[4] [oneAPI](https://www.oneapi.com/)

[5] [CUDA](https://developer.nvidia.com/cuda-zone)
