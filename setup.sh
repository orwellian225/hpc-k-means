#!/bin/sh

mkdir build
mkdir data
mkdir -p results/images

git submodule update --init --recursive
conda init
conda activate
cmake -S . -B build/