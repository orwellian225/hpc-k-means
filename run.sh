#!/bin/sh

num_points=1000
num_classes=10
dimension=10

data_filepath="data/${num_points}_${dimension}D.csv"
serial_out_filepath="results/${num_points}_${dimension}D_${num_classes}C_serial.csv"
mpi_out_filepath="results/${num_points}_${dimension}D_${num_classes}C_mpi.csv"
cuda_out_filepath="results/${num_points}_${dimension}D_${num_classes}C_cuda.csv"

python3 scripts/generate_data.py $dimension $num_points data/

echo "Execution times:"
serial_time=$(build/kmeans-serial $dimension $num_points $data_filepath $serial_out_filepath $num_classes 1000 0)
echo "\tSerial: ${serial_time} ms"

mpi_time=$(mpiexec -np 2 build/kmeans-mpi $dimension $num_points $data_filepath $mpi_out_filepath $num_classes 1000 0)
echo "\tMPI: ${mpi_time} ms"

cuda_time=$(build/kmeans-cuda $dimension $num_points $data_filepath $cuda_out_filepath $num_classes 1000 0)
echo "\tCUDA: ${cuda_time} ms"

echo "Output files:"
echo "\tSerial: ${serial_out_filepath}"
echo "\tMPI: ${mpi_out_filepath}"
echo "\tCUDA: ${cuda_out_filepath}"

python3 scripts/class_loss.py $num_points $num_classes $serial_out_filepath $mpi_out_filepath $cuda_out_filepath
python3 scripts/class_error.py $num_points $num_classes $serial_out_filepath $mpi_out_filepath $cuda_out_filepath

echo "${dimension},${num_points},${num_classes},${data_filepath},${serial_time},${mpi_time},${cuda_time},${serial_out_filepath},${mpi_out_filepath},${cuda_out_filepath}" >> cumulative_results.csv
