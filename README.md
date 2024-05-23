# COMS4040A HPC Project

## Quick Execution

```bash
./setup.sh
./compile.sh
./run.sh
```

## Execution

### Dependencies

* CUDA - To run the CUDA implementation
* MPI - To run the MPI implementation
* CMake - To compile any implementation
* fmtlib - To compile any implementation
* Python3 - To run support scripts

### Setup

```bash
git submodule update --init --recursive
mkdir -p build  data/images results/images
cmake -S . -B build/
```

### Compile

```bash
cd build
make
```

### Generating Test Data

Will generate $d$ Perlin Noise textures of size $n \times n$ and then merge the textures into a single data set where each texture forms one dimension.

```bash
python3 scripts/generate_data.py <data dimension> <number of points> data/
```

> - $n$ is the number of points
> - $d$ is the number of dimensions

### Running

All implementations take the same arguments:

Binary names:
* Serial: `kmeans-serial`
* MPI: `kmeans-mpi`
* CUDA: `kmeans-cuda`

```bash
<implementation-binary> <data dimension> <number of points> <data file> <output file> <maximum iterations> <random seed>
```

> The MPI binary must be executed with mpiexec / mpirun in the following manner:
>
> ```bash
> mpiexec -np <number of processors> <normal binary execution and args>
> ```

## Data

* $n$ points
* $d$ dimensionality
* $k$ classes
* Basis vectors $\hat{e}_1, \hat{e}_2, ..., \hat{e}_d$

### Input

* px -> point x
* ex -> $\hat{e}_x$

```csv
<p1 e1 value>,<p1 e2 value>, ..., <p1 e_d value>
<p2 e1 value>,<p2 e2 value>, ..., <p2 e_d value>
...
<pn e1 value>,<pn e2 value>, ..., <pn e_d value>
```

### Output

* px -> point x
* cy -> centroid y
* ez -> $\hat{e}_z$
* pxi -> point x index = x
* cyi -> centroid y index = y

```csv
<p1 e1 value>,<p1 e2 value>, ..., <p1 e_d value>
<p2 e1 value>,<p2 e2 value>, ..., <p2 e_d value>
...
<pn e1 value>,<pn e2 value>, ..., <pn e_d value>


<c1 e1 value>,<c1 e2 value>, ..., <c1 e_d value>
<c2 e1 value>,<c2 e2 value>, ..., <c2 e_d value>
...
<ck e1 value>,<ck e2 value>, ..., <ck e_d value>

<p1i>,<cyi>
<p2i>,<cyi>
...
<pni>,<cyi>
```