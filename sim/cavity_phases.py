# Copyright (C) 2026 Wolfgang Flachberger
# SPDX-License-Identifier: GPL-3.0-or-later
"""Unsteady Stokes flow in a lid-driven square cavity: three phases in one computation.

  A  lid started from rest        driven, non-stationary   (no extremal statement about P)
  B  driven stationary state      P = W_lid, P at Helmholtz's minimum P_S
  C  lid stopped, closed decay    Kdot + P = 0, constrained maximum of P

Unit square, rho = eta = 1, lid velocity U*16 x^2 (1-x)^2 (regularized: finite dissipation at the corners),
Taylor-Hood P2/P1, backward Euler (the incremental minimization of the paper's Sec. 3.2),
Rayleighian with (1/2) eta |grad u|^2, i.e. Stokes in Laplacian form.

Run:  docker run --rm -v $PWD:/work -w /work --user $(id -u):$(id -g) -e HOME=/tmp \
          dolfinx/dolfinx:v0.11.0 python3 cavity_phases.py [N] [dt]
"""
import sys, time
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl, basix.ufl
from dolfinx import mesh, fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
dt_val = float(sys.argv[2]) if len(sys.argv) > 2 else 5e-4
rho, eta, U = 1.0, 1.0, 1.0
T_off, T_end = 0.20, 0.40           # lid stops at T_off (start of phase C)

comm = MPI.COMM_WORLD
msh = mesh.create_unit_square(comm, N, N, mesh.CellType.triangle)
P2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(2,))
P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
W = fem.functionspace(msh, basix.ufl.mixed_element([P2, P1]))
V, _ = W.sub(0).collapse()
Q, _ = W.sub(1).collapse()

fdim = msh.topology.dim - 1
lid_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
wall_facets = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0))
facet_tags = mesh.meshtags(msh, fdim, lid_facets, np.full(len(lid_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
dx = ufl.dx

u_lid = fem.Function(V)
u_lid.interpolate(lambda x: np.vstack((U * 16 * x[0]**2 * (1 - x[0])**2, np.zeros(x.shape[1]))))
u_zero = fem.Function(V)
p_zero = fem.Function(Q)
lid_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, lid_facets)
wall_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
bcs = [fem.dirichletbc(u_zero, wall_dofs, W.sub(0)),
       fem.dirichletbc(u_lid, lid_dofs, W.sub(0)),      # applied last: wins at the corners (both are zero there)
       fem.dirichletbc(p_zero, p_dofs, W.sub(1))]       # pin the pressure constant

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
w_n = fem.Function(W)                    # previous time level
u_n, _ = ufl.split(w_n)
w_h = fem.Function(W)                    # current solution
u_h, p_h = ufl.split(w_h)
dt = fem.Constant(msh, PETSc.ScalarType(dt_val))
f0 = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

a_visc = eta * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - p * ufl.div(v) * dx - q * ufl.div(u) * dx
a_be = (rho / dt) * ufl.inner(u, v) * dx + a_visc
L_be = (rho / dt) * ufl.inner(u_n, v) * dx
a_st = a_visc
L_st = ufl.inner(f0, v) * dx

def lu_solver(a_form):
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    return ksp

def solve(ksp, a_form, L_form, target):
    b = assemble_vector(L_form)
    apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    ksp.solve(b, target.x.petsc_vec)
    target.x.scatter_forward()
    b.destroy()

w_lid_ext = fem.Function(W)                       # lid velocity extended by zero into the mixed space
w_lid_ext.x.array[lid_dofs[0]] = u_lid.x.array[lid_dofs[1]]

n_vec = ufl.FacetNormal(msh)
def diagnostics(w):
    uu, _ = ufl.split(w)
    K = fem.assemble_scalar(fem.form(0.5 * rho * ufl.inner(uu, uu) * dx))
    P = fem.assemble_scalar(fem.form(eta * ufl.inner(ufl.grad(uu), ufl.grad(uu)) * dx))
    Wl = fem.assemble_scalar(fem.form(eta * ufl.inner(uu, ufl.dot(ufl.grad(uu), n_vec)) * ds(1)))
    return K, P, Wl

# --- steady Stokes state (phase B reference, Helmholtz minimum)
a_st_form, L_st_form = fem.form(a_st), fem.form(L_st)
w_S = fem.Function(W)
solve(lu_solver(a_st_form), a_st_form, L_st_form, w_S)
K_S, P_S, W_S = diagnostics(w_S)
Wc_S = fem.assemble_scalar(fem.form(ufl.action(ufl.action(a_st, w_S) - L_st, w_lid_ext)))   # consistent lid power, steady
u_S, _ = ufl.split(w_S)
Pdiff_form = fem.form(eta * ufl.inner(ufl.grad(u_h - u_S), ufl.grad(u_h - u_S)) * dx)   # P of the difference field
cross_form = fem.form(2 * eta * ufl.inner(ufl.grad(u_S), ufl.grad(u_h - u_S)) * dx)     # cross term (should vanish)
numdiss_form = fem.form(0.5 * rho * ufl.inner(u_h - u_n, u_h - u_n) * dx)               # backward-Euler numerical dissipation
Wc_form = fem.form(ufl.action(ufl.action(a_be, w_h) - L_be, w_lid_ext))                  # consistent lid power (discrete flux)

# --- transient: backward Euler, matrix factorized once (Dirichlet rows do not change when the lid stops)
a_be_form, L_be_form = fem.form(a_be), fem.form(L_be)
ksp = lu_solver(a_be_form)
nsteps = int(round(T_end / dt_val))
rec = np.zeros((nsteps, 9))
t, lid_on = 0.0, True
t0 = time.time()
for n in range(nsteps):
    t = (n + 1) * dt_val
    if lid_on and t > T_off + 1e-12:
        u_lid.x.array[:] = 0.0
        lid_on = False
    solve(ksp, a_be_form, L_be_form, w_h)
    K, P, Wl = diagnostics(w_h)
    Pd = fem.assemble_scalar(Pdiff_form)
    cr = fem.assemble_scalar(cross_form)
    nd = fem.assemble_scalar(numdiss_form)
    Wc = fem.assemble_scalar(Wc_form) if lid_on else 0.0
    rec[n] = (t, K, P, Wl, Pd, cr, nd, float(lid_on), Wc)
    w_n.x.array[:] = w_h.x.array
wall = time.time() - t0

# --- checks
t_, K_, P_, W_, Pd_, cr_, nd_, on_, Wc_ = rec.T
A_mask = on_ > 0.5
dev = (P_ - P_S) / P_S
i_B = np.argmax((dev < 1e-3) & A_mask)          # first step within 0.1 % of the Helmholtz minimum
t_B = t_[i_B]
i_off = np.argmax(~A_mask)                      # first step of phase C
K_off = K_[i_off - 1]
# discrete energy identity per step: (K^{n+1}-K^n)/dt + P^{n+1} + nd^{n+1}/dt = W^{n+1}
Kprev = np.concatenate(([0.0], K_[:-1]))
resid = (K_ - Kprev) / dt_val + P_ + nd_ / dt_val - Wc_
diss_C = dt_val * P_[~A_mask].sum()
numdiss_C = nd_[~A_mask].sum()
print(f"N={N} dt={dt_val} dofs={W.dofmap.index_map.size_global * W.dofmap.index_map_bs} steps={nsteps} wall={wall:.1f}s")
print(f"steady Stokes: K_S={K_S:.6e} P_S={P_S:.6e} pointwise lid power W_S={W_S:.6e} ((P_S-W_S)/P_S={(P_S - W_S) / P_S:.2e}); consistent lid power Wc_S={Wc_S:.6e} ((P_S-Wc_S)/P_S={(P_S - Wc_S) / P_S:.2e})")
print(f"phase A: P(t) >= P_S ? min(P-P_S)/P_S = {dev[A_mask].min():.2e}; max increase of P per step = {np.diff(P_[A_mask]).max():.2e}")
print(f"         cross term max |2 eta <grad u_S, grad w>| / P_S = {np.abs(cr_[A_mask]).max() / P_S:.2e}")
print(f"         first step within 0.1% of P_S: t_B = {t_B:.4f}; at lid-off (t={t_[i_off]:.4f}) (P-P_S)/P_S = {dev[i_off - 1]:.2e}, (P-Wc)/P = {(P_[i_off - 1] - Wc_[i_off - 1]) / P_[i_off - 1]:.2e}, pointwise (P-W)/P = {(P_[i_off - 1] - W_[i_off - 1]) / P_[i_off - 1]:.2e}")
print(f"phase C: K at lid-off = {K_off:.6e}; sum dt*P over C = {diss_C:.6e}; BE numerical dissipation = {numdiss_C:.3e}; K_end = {K_[-1]:.3e}")
print(f"         closure: K_off - (diss_C + numdiss_C + K_end) = {K_off - diss_C - numdiss_C - K_[-1]:.2e}")
print(f"discrete energy identity (K^n-K^(n-1))/dt + P^n + numdiss^n/dt - Wc^n: max |residual| = {np.abs(resid).max():.2e}  (relative to P_S: {np.abs(resid).max() / P_S:.2e})")
print(f"phase C: share of K_off dissipated in the first step after the lid stop (impulsive-stop boundary layer): {(dt_val * P_[i_off] + nd_[i_off]) / K_off:.3f}; BE numerical share over C: {numdiss_C / K_off:.3f}")
print(f"phase A: first-step values t={t_[0]:.2e} K={K_[0]:.3e} P={P_[0]:.3e} Wc={Wc_[0]:.3e}; at t_B: K={K_[i_B]:.4e} P={P_[i_B]:.4e}")
# late-time decay rate in C (P ~ exp(-2 lambda_1 t))
sel = (~A_mask) & (t_ > T_off + 0.1)
lam = -0.5 * np.polyfit(t_[sel], np.log(P_[sel]), 1)[0]
print(f"phase C: fitted slowest Stokes eigenvalue lambda_1 = {lam:.3f}")
np.savez(f"cavity_phases_N{N}_dt{dt_val:g}.npz", rec=rec, K_S=K_S, P_S=P_S, W_S=W_S, Wc_S=Wc_S, t_B=t_B, T_off=T_off, N=N, dt=dt_val)
