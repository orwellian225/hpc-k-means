#!/bin/sh

# $1: Image name
# $2: Image location
# $3: Image size
# $4: Number of clusters

num_points=$(($3*$3))
file_name="${1}_${num_points}_3D_image.csv"

python3 scripts/image_to_data.py $1 $2 data/
build/kmeans-serial 3 $num_points data/$file_name results/$file_name $4 10000
python3 scripts/results_to_image.py $1 results/$file_name results/images/