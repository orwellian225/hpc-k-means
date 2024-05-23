#!/bin/sh

mkdir build
mkdir -p data/images
mkdir -p results/images

git submodule update --init --recursive

cmake -S . -B build/