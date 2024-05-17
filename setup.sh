#!/bin/sh

mkdir build
mkdir -p data/images
mkdir -p results/images

git submodule update --init --recursive

conda init
conda activate
conda install cmake

cmake -S . -B build/