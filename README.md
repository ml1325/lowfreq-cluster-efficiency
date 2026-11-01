# lowfreq-cluster-efficiency

Code and data for the numerical tests in

M. Lou, Y. Zhang, Z. Tao, *A low-frequency refinement of a posteriori efficiency bounds for eigenvalue clusters*.

All computations use conforming P1 finite elements on structured meshes, equilibrated RT1 fluxes and P2 local residual liftings as in Cancès et al. (2017). The exact eigenpairs are used only for the errors and the exact angle. All numbers are floating-point values.

## Running

Python 3 with NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`).

```
python test1.py      # Test 1, writes results/test1.csv (about 3 min)
python test2.py      # Test 2, writes results/test2.csv and results/test2_p.csv (about 10 min)
python plot_fig1.py  # Figure 1 from results/test1.csv
```

`python test1.py 80` and `python test2.py 80` stop at the second mesh.

## Files

- `fem.py`: meshes, P1 assembly, eigenvalue solves, Crouzeix–Raviart lower bounds, quadrature, exact eigenpairs
- `estimators.py`: equilibrated fluxes, residual liftings, cluster errors, the exact angle
- `test1.py`: single eigenvalues on the triangle with vertices (0,0), (1,0), (0,1)
- `test2.py`: the cluster {λ10, λ11, λ12} on the rectangle (0,1.1) × (0,1)
- `plot_fig1.py`: Figure 1

## Results

`results/test1.csv`, one row per mesh `N` and eigenvalue `i`:
`cbar` is the factor c̄_ih of Eq. (12), `I_ub` the upper-bound effectivity index of Cancès et al. (2017), `ub_orig` the improved upper bound of Eq. (12), and `I_orig`, `I_L2`, `I_dbar`, `I_dhat` the lower-bound indices of Eq. (12), Proposition 1 and Corollary 1 with the a priori and a posteriori angle (`inf` = void bound). `dbar`, `dhat` and `delta` are the a priori bound, the a posteriori bound and the exact value of the angle for p = i − 1.

`results/test2.csv`, one row per mesh `Ny` (Table 1):
`ratio_low` and `ratio_up` bound ‖Res‖²/(E² + Q_h) from below (liftings) and above (fluxes), `const_orig`, `const_L2` and `const_dbar` are the efficiency constants of Eq. (5), Proposition 1 and Theorem 1, `ell` is ℓ_h, `dhat` and `const_dhat` refer to Theorem 1(iv) with p = 9, `best_p` and `const_best` to the best p, `lv_low` and `lv_up` to the bound of Liu and Vejchodský (2022), and `sigma_min` is the smallest singular value in the check of (A2).

`results/test2_p.csv` lists, for p = 1, ..., 9, the a posteriori angle `dhat`, the first bound `sin_theta` in the proof of Theorem 1(iv), the exact angle `delta`, `C_p` and the constant `C_p/(1 − dhat²)`.

## License

MIT
