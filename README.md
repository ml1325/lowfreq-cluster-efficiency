# Low-frequency refinement of efficiency bounds for eigenvalue clusters

Code, numerical output, and Figure 1 accompanying the manuscript:

> M. Lou, Y. Zhang, and Z. Tao, *A low-frequency refinement of a posteriori efficiency bounds for eigenvalue clusters*.

The experiments compare the original residual efficiency bounds of Cancès et al., an elementary $L^2$ comparison, and the angle-dependent bounds in the manuscript. They cover individual Laplace eigenvalues on a right triangle and eigenvalue clusters on a rectangle. The repository also contains the computations for the a posteriori angle bound, the comparison with Liu and Vejchodský, and the assumption and sensitivity checks.

## Start here

- Read the supplied numerical tables in [`results/`](results/) without running a solver.
- Open [Figure 1 as a vector PDF](figures/fig_set1.pdf).
- Run the two short diagnostic scripts below to check an installation.
- Use the reproduction commands to recompute the experiments into a separate directory.

The archived data use mesh levels up to $N=320$ or $N_y=320$. A run with $N=640$ is not included.

## Numerical setting

Test 1 uses the triangle with vertices $(0,0)$, $(1,0)$, and $(0,1)$. It compares the bounds for the first four eigenvalues. Test 2 uses the rectangle $(0,1.1)\times(0,1)$ and includes the high cluster $\{10,11,12\}$ and the low cluster $\{2,3\}$.

The computations use conforming $P_1$ finite elements on structured right-isosceles meshes. Here $h$ is the largest element diameter. The main experiments use equilibrated $RT_1$ fluxes and $P_2$ local residual liftings. Additional reconstruction degrees are used in the sensitivity checks. Algebraic eigenpairs are solved to high accuracy and their residuals are recorded; the analysis in the manuscript assumes exact Galerkin eigenpairs.

Analytic eigenpairs are used to evaluate the errors and the exact approximation angle in these benchmark problems. They are not inputs to the computable a posteriori angle majorant. The reported quantities are floating-point evaluations, rather than interval-arithmetic enclosures. In particular, the overlap-matrix check of the non-orthogonality assumption is numerical.

## Installation

The scripts require Python 3, NumPy, and SciPy. Matplotlib is needed to generate Figure 1. Dependencies are listed in [`requirements.txt`](requirements.txt).

```bash
git clone https://github.com/ml1325/lowfreq-cluster-efficiency.git
cd lowfreq-cluster-efficiency
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONUTF8 = "1"
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The UTF-8 setting helps when redirecting output containing mathematical symbols on Windows. Run the commands below from the repository root.

## Installation checks

```bash
python fem_core.py
python test_est.py
```

`fem_core.py` prints finite-element eigenvalues, analytic eigenvalues, algebraic residuals, Crouzeix-Raviart lower-bound checks, and analytic Rayleigh quotients. `test_est.py` prints flux and lifting diagnostics for eigenvalues 1, 3, and 4 on a small triangular mesh. Inspect the numerical diagnostics alongside the process exit code.

## Reproduce the manuscript results

The committed `results/` directory contains the output used for the manuscript. The commands here write to `reproduced/`, so the archived output remains available for comparison.

Create the output directories first. The experiment scripts expect them to exist.

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force reproduced/out1, reproduced/out2, reproduced/out_delta, reproduced/out_lv, reproduced/out3, reproduced/figures | Out-Null
```

Linux or macOS:

```bash
mkdir -p reproduced/out1 reproduced/out2 reproduced/out_delta reproduced/out_lv reproduced/out3 reproduced/figures
```

| Manuscript item | Command | Main output |
| --- | --- | --- |
| Test 1: original bounds, the elementary $L^2$ route, and Corollary 1 with the a priori angle bound | `python set1.py 320 reproduced/out1` | `set1_rows.json`, `set1_table.md` |
| Test 1: Corollary 1 with the a posteriori angle bound | `python delta_post_test1.py reproduced/out1/set1_rows.json reproduced/out_delta/test1_deltahat.md` | `test1_deltahat.md` |
| Figure 1 from the recomputed Test 1 data | `python make_fig1.py reproduced reproduced/figures` | `fig_set1.pdf` |
| Test 2: cluster efficiency constants and Table 1 | `python set2.py 320 reproduced/out2` | `set2_rows.json`, `set2_table.md` |
| Test 2: a posteriori angle estimates and optimization over $p$ | `python exp_delta_post.py 320 reproduced/out_delta` | `delta_post_rows.json`, `delta_post_table.md` |
| Test 2: comparison with Liu-Vejchodský, Section 5.3, Eq. (45) | `python exp_lv_compare.py 320 reproduced/out_lv` | `lv_compare_rows.json`, `lv_compare_table.md` |
| Additional checks: coarse meshes, overlap matrices, and reconstruction sensitivity | `python exp_cheap.py reproduced/out3` | Three JSON files described below |

For a smaller initial run, replace `320` with `40` in an individual solver command. The scripts use the predefined mesh levels no larger than the supplied maximum. Figure 1 expects all four Test 1 mesh levels and both Test 1 output files, so generate the full Test 1 data before plotting a new figure.

To regenerate Figure 1 directly from the archived results, without rerunning a finite-element computation:

```bash
python make_fig1.py results reproduced/figures
```

The plotting script writes the vector PDF used for the publication figure. Runtime and memory use depend on the mesh, sparse solver, and machine; the diagnostic scripts are much smaller than the finest-mesh experiments.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `fem_core.py` | Mesh generation, $P_1$ assembly, eigenvalue solves, Crouzeix-Raviart lower bounds, quadrature, and analytic eigenfunctions |
| `estimators.py` | Equilibrated fluxes, local residual liftings, cluster error quantities, exact approximation angles, and fine-space residual liftings |
| `liftings_pk.py` | Higher-degree local liftings and collapsed Gauss quadrature for reconstruction-sensitivity checks |
| `set1.py`, `set2.py` | The two main numerical experiments |
| `delta_post_test1.py` | Test 1 a posteriori angle majorants from the recorded residual data |
| `exp_delta_post.py` | Test 2 angle bounds and the comparison of admissible choices of $p$ |
| `exp_lv_compare.py` | Operator-norm comparison using residual Gram matrices and a linear flux choice |
| `exp_cheap.py` | Coarse-grid tests, overlap-matrix checks, and the Test 1 reconstruction-sensitivity experiment |
| `test_est.py` | Small flux and lifting diagnostic |
| `make_fig1.py` | Reproducible plotting of Figure 1 |
| `results/` | Archived JSON rows, Markdown tables, and console logs |
| `figures/fig_set1.pdf` | Vector version of Figure 1 |

The additional results in `results/out3/` are:

- `set2_coarse_rows.json`: Test 2 with $N_y=10,20$;
- `assumption26_rows.json`: singular values of the exact-discrete overlap matrices;
- `set1_N40_sensitivity.json`: sensitivity to flux degree, lifting degree, and quadrature for Test 1 at $N=40$.

## Reading the output

- `orig` means the original bounds of Cancès et al.; `L2` is the elementary comparison in Proposition 1; `new` is the angle-dependent bound. Equation numbers in code comments can refer to the source papers, rather than the manuscript.
- In Test 1, `aux="paper"` uses the published auxiliary eigenvalue lower bounds. `aux="CR46"` records the alternative Crouzeix-Raviart input. The figure uses the `paper` rows.
- The lower-bound effectivity index is $I^{\mathrm{lb}}=(\lambda_{ih}-\lambda_i)/\widetilde\eta_i^2$. A zero lower bound gives no improvement and is reported as void. The JSON output may contain `Infinity` for such indices; Python's JSON reader accepts it, while stricter JSON readers may need explicit handling.
- In Test 2, the residual ratios are relative to $E^2+Q_h$. The column `cbar` is the coefficient of $E^2$ in the original bound. The effective constant in the manuscript table is $(\bar c_h E^2+Q_h)/(E^2+Q_h)$.
- The elementary cluster constant $1/(1-\ell_h)$ applies only when `loss` $=\ell_h<1$. A void entry means this condition fails.
- `chk_*` fields record numerical inequality checks. The Markdown tables summarize the checks selected by each script. Fine-space liftings and fluxes provide the two sides of the reported residual-norm comparisons, evaluated in floating point.
- Small differences between reruns can arise from numerical library versions and floating-point arithmetic. Compare the quantities and recorded residuals, rather than expecting byte-identical output or identical wall-clock timings.

## References and citation

Please cite the accompanying manuscript when using these experiments and record the Git commit used for a reproduction.

The principal source references are:

1. E. Cancès, G. Dusson, Y. Maday, B. Stamm, and M. Vohralík, *Guaranteed and robust a posteriori bounds for Laplace eigenvalues and eigenvectors: conforming approximations*, SIAM Journal on Numerical Analysis 55 (2017), 2228-2254. [DOI](https://doi.org/10.1137/15M1038633).
2. The same authors, *Guaranteed a posteriori bounds for eigenvalues and eigenvectors: multiplicities and clusters*, Mathematics of Computation 89 (2020), 2563-2611. [DOI](https://doi.org/10.1090/mcom/3549).
3. X. Liu and T. Vejchodský, *Fully computable a posteriori error bounds for eigenfunctions*, Numerische Mathematik 152 (2022), 183-221. [DOI](https://doi.org/10.1007/s00211-022-01304-0).
4. C. Carstensen and J. Gedicke, *Guaranteed lower bounds for eigenvalues*, Mathematics of Computation 83 (2014), 2605-2629. [DOI](https://doi.org/10.1090/S0025-5718-2014-02833-0).

## License

MIT. See [`LICENSE`](LICENSE).
