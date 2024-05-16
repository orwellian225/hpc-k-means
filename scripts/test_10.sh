#!/bin/sh

# $1 number of classes
# $2 number of iterations

build/kmeans-serial 2 10 data/10_2D.csv results/10_2D_${1}C_serial.csv $1 $2 0
mpiexec -np 2 build/kmeans-mpi 2 10 data/10_2D.csv results/10_2D_${1}C_mpi.csv $1 $2 0
build/kmeans-cuda 2 10 data/10_2D.csv results/10_2D_${1}C_cuda.csv $1 $2 0

python3 scripts/class_error.py 10 $1 results/10_2D_${1}C_serial.csv results/10_2D_${1}C_mpi.csv results/10_2D_${1}C_cuda.csv
python3 scripts/class_loss.py 10 $1 results/10_2D_${1}C_serial.csv results/10_2D_${1}C_mpi.csv results/10_2D_${1}C_cuda.csv