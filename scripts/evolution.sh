#!/bin/bash

python3 scripts/generate_data.py 3 512 data/

build/kmeans-serial 3 262144 data/262144_3D.csv results/262144_3D_2C.csv 2 1000 0
build/kmeans-serial 3 262144 data/262144_3D.csv results/262144_3D_5C.csv 5 1000 0
build/kmeans-serial 3 262144 data/262144_3D.csv results/262144_3D_10C.csv 10 1000 0
build/kmeans-serial 3 262144 data/262144_3D.csv results/262144_3D_100C.csv 100 1000 0

python3 scripts/data_to_image.py random_original data/262144_3D.csv data/images/
python3 scripts/results_to_image.py random_2C results/262144_3D_2C.csv results/images/ ""
python3 scripts/results_to_image.py random_5C results/262144_3D_5C.csv results/images/ ""
python3 scripts/results_to_image.py random_10C results/262144_3D_10C.csv results/images/ ""
python3 scripts/results_to_image.py random_100C results/262144_3D_100C.csv results/images/ ""