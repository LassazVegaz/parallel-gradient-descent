#PBS -l walltime=00:01:00
#PBS -l nodes=1
#PBS -j oe
#PBS -N omp

cd $PBS_O_WORKDIR
cat $PBS_NODEFILE

./omp.o
