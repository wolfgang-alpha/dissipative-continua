# Variational principles for dissipative continua

LaTeX sources, simulation code and data for

> W. Flachberger and L. Einkemmer, *Variational principles for dissipative continua: transport theorems, inertia, and the minimum–maximum duality*, submitted to Physics of Fluids. Preprint: [arXiv:2611.12045](https://arxiv.org/abs/2611.12045).

## Contents

| Path | Contents |
|---|---|
| `main.tex`, `fig-cavity-phases.pdf` | Preprint version (article class) |
| `pof/` | Journal version (AIP REVTeX 4.2): `main.tex`, `refs.bib`, `aipnum4-2-pof.bst`, figures |
| `pof/fig-regimes.tex` | Source of Fig. 2 (a standalone TikZ file; the compiled `fig-regimes.pdf` is included) |
| `sim/` | Lid-driven cavity computation behind Fig. 1: solver, plotting script, results |

## Building the paper

Both versions build with a standard TeX Live installation.

```sh
# preprint
pdflatex main.tex && pdflatex main.tex

# journal version
cd pof && latexmk -pdf main.tex
```

`pof/aipnum4-2-pof.bst` is the REVTeX `aipnum4-2` style with issue numbers added after the volume, as Physics of Fluids requests.

## Reproducing Fig. 1

Unsteady Stokes flow in the unit square with lid velocity `16 x^2 (1-x)^2` and ρ = η = U = 1, computed with Taylor–Hood P2/P1 elements and backward Euler. The lid is stopped at t = 0.2. The solver uses FEniCSx (DOLFINx v0.11.0) and runs in the official container:

```sh
cd sim
docker run --rm -v $PWD:/work -w /work --user $(id -u):$(id -g) -e HOME=/tmp \
    dolfinx/dolfinx:v0.11.0 python3 cavity_phases.py 64 0.000125
```

This writes `cavity_phases_N64_dt0.000125.npz` and prints the checks quoted in the paper: the Helmholtz minimum P_S, the discrete energy identity, and the slowest Stokes eigenvalue. Plotting needs NumPy, Matplotlib and the STIX fonts:

```sh
python3 plot_phases.py cavity_phases_N64_dt0.000125.npz                                  # ../fig-cavity-phases.pdf (16 cm, preprint)
python3 plot_phases.py cavity_phases_N64_dt0.000125.npz 13.8 ../pof/fig-cavity-phases.pdf  # journal version
```

The other `.npz` files are resolution checks: N = 64 with Δt = 5·10⁻⁴ and 2.5·10⁻⁴, and N = 96 with Δt = 5·10⁻⁴.

### Data format

Each `.npz` file holds the array `rec` with one row per time step. Its columns are:

1. time t
2. kinetic energy K
3. dissipation P
4. pointwise lid power
5. dissipation of the difference to the steady Stokes field
6. cross term (vanishes by orthogonality)
7. numerical dissipation of the backward-Euler step
8. lid-on flag
9. consistent lid power

It also holds the scalars `K_S`, `P_S` (steady Stokes values), `W_S`, `Wc_S` (steady lid power), `t_B` (time to within 0.1 % of P_S), `T_off`, `N` and `dt`.
