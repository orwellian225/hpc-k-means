# COMS4040A HPC Project

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