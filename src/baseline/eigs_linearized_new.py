# --- file: eigs_linearized.py ---
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import ufl
from dolfinx import fem
from petsc4py import PETSc
from slepc4py import SLEPc

# Version-robust assembly imports
try:
    from dolfinx.fem.petsc import assemble_matrix
except Exception:
    from dolfinx.fem import assemble_matrix


def _owned_global_dofs(V: fem.FunctionSpace) -> np.ndarray:
    """Owned (global) dof ids for scalar space V."""
    im = V.dofmap.index_map
    start, end = im.local_range
    return np.arange(start, end, dtype=np.int32)


def _to_global_dofs(V: fem.FunctionSpace, local_dofs: np.ndarray) -> np.ndarray:
    """Map local owned dof ids -> global ids."""
    start, _ = V.dofmap.index_map.local_range
    return (start + local_dofs.astype(np.int64)).astype(np.int32)


def _vec_data(fun: fem.Function) -> np.ndarray:
    """Writable coefficient array (owned+ghost)."""
    try:
        return fun.vector.array
    except AttributeError:
        return fun.x.array


def _ghost_forward(fun: fem.Function) -> None:
    try:
        fun.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    except AttributeError:
        if hasattr(fun.x, "scatter_forward"):
            fun.x.scatter_forward()


def _is_complex_petsc() -> bool:
    return np.dtype(PETSc.ScalarType).kind == "c"


@dataclass
class Blocks:
    # reduced sizes
    n_free: int
    # full (global) sizes for the shared CG1 layout (omega == psi)
    n_full: int

    # reduced blocks
    A11_sub: PETSc.Mat          # (n_free x n_free)
    A12_row: PETSc.Mat          # (n_free x n_full)
    Moo_sub: PETSc.Mat          # (n_free x n_free)

    # full operators
    Mpsi: PETSc.Mat             # (n_full x n_full)
    K: PETSc.Mat                # (n_full x n_full) pinned

    # embedding index list (GLOBAL indices for free omega dofs owned on this rank)
    free_idx: np.ndarray        # length == local n_free on rank in your construction


def assemble_reduced_blocks(
    mesh,
    Vomega: fem.FunctionSpace,
    Vpsi: fem.FunctionSpace,
    Omega_star: fem.Function,
    psi_star: fem.Function,
    mu: float,
    eps: float,
    facet_tags,
    outer_tag: int = 1,
) -> Blocks:
    """
    Uses exactly your current discretization conventions:

      A11:  eps*∫ ∇u·∇w + ∫ (vstar·∇u) w  - ∫ u w
      A12:  ∫ (∇^⊥phi)·∇Ω*  w
      K:    ∫ ∇phi · ∇phi_t   (Neumann natural BC) + one pinned gauge dof
      Poisson coupling in operator application uses: z = K^{-1} (Mpsi * omega_full)

    IMPORTANT: assumes omega and psi share the same scalar CG1 dof layout (same global size).
    """
    comm = mesh.comm

    # sanity: same layout
    n_o = int(Vomega.dofmap.index_map.size_global) * int(getattr(Vomega.dofmap.index_map, "block_size", 1))
    n_p = int(Vpsi.dofmap.index_map.size_global) * int(getattr(Vpsi.dofmap.index_map, "block_size", 1))
    if n_o != n_p:
        raise RuntimeError("Vomega and Vpsi must have identical global dof counts (shared CG1 layout).")

    w = ufl.TestFunction(Vomega)
    u = ufl.TrialFunction(Vomega)
    phi_t = ufl.TestFunction(Vpsi)
    phi = ufl.TrialFunction(Vpsi)

    xcoord = ufl.SpatialCoordinate(mesh)

    gpsi = ufl.grad(psi_star)
    vstar = ufl.as_vector((-gpsi[1], gpsi[0])) - mu * ufl.as_vector((xcoord[0], xcoord[1]))
    grad_Os = ufl.grad(Omega_star)

    a11_form = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        + ufl.inner(vstar, ufl.grad(u)) * w * ufl.dx
        - ufl.inner(u, w) * ufl.dx
    )

    gphi = ufl.grad(phi)
    rot_phi = ufl.as_vector((-gphi[1], gphi[0]))
    a12_form = ufl.inner(rot_phi, grad_Os) * w * ufl.dx

    mass_omega_form = u * w * ufl.dx
    mass_psi_form = phi * phi_t * ufl.dx

    # Your current convention
    K_form = ufl.inner(-ufl.grad(phi), ufl.grad(phi_t)) * ufl.dx

    A11 = assemble_matrix(fem.form(a11_form)); A11.assemble()
    A12 = assemble_matrix(fem.form(a12_form)); A12.assemble()
    Moo = assemble_matrix(fem.form(mass_omega_form)); Moo.assemble()
    Mpsi = assemble_matrix(fem.form(mass_psi_form)); Mpsi.assemble()
    K = assemble_matrix(fem.form(K_form)); K.assemble()

    outer_facets = facet_tags.find(int(outer_tag))

    # Dirichlet dofs for omega (outer boundary)
    bdofs_o_local = fem.locate_dofs_topological(Vomega, mesh.topology.dim - 1, outer_facets)
    bdofs_o_glob = _to_global_dofs(Vomega, bdofs_o_local)

    all_o_glob = _owned_global_dofs(Vomega)
    if bdofs_o_glob.size:
        bset = set(int(i) for i in bdofs_o_glob)
        mask = np.array([int(g) not in bset for g in all_o_glob], dtype=bool)
    else:
        mask = np.ones(all_o_glob.size, dtype=bool)

    free_glob = all_o_glob[mask].astype(np.int32)
    is_free = PETSc.IS().createGeneral(free_glob, comm=comm)

    # all psi owned dofs (for column restriction of A12)
    all_p_glob = _owned_global_dofs(Vpsi).astype(np.int32)
    is_all_psi = PETSc.IS().createGeneral(all_p_glob, comm=comm)

    # Gauge pin for Neumann K
    bdofs_p_local = fem.locate_dofs_topological(Vpsi, mesh.topology.dim - 1, outer_facets)
    if bdofs_p_local.size > 0:
        pin_dof = int(_to_global_dofs(Vpsi, bdofs_p_local[:1])[0])
    else:
        pin_dof = int(all_p_glob[0])

    K.zeroRowsColumns([pin_dof], diag=1.0)

    # Restricted blocks
    A11_sub = A11.createSubMatrix(is_free, is_free)
    Moo_sub = Moo.createSubMatrix(is_free, is_free)
    A12_row = A12.createSubMatrix(is_free, is_all_psi)

    n_free = int(Moo_sub.getSize()[0])
    n_full = n_o

    return Blocks(
        n_free=n_free,
        n_full=n_full,
        A11_sub=A11_sub,
        A12_row=A12_row,
        Moo_sub=Moo_sub,
        Mpsi=Mpsi,
        K=K,
        free_idx=free_glob,
    )


class _AredShell:
    """
    y = A_red x = A11_sub x + A12_row * ( K^{-1} * (Mpsi * P x) )
    """

    def __init__(self, blocks: Blocks, comm, ksp_prefix: str = "poisson_"):
        self.b = blocks
        self.comm = comm

        # Poisson solver (K)
        self.kspK = PETSc.KSP().create(comm)
        self.kspK.setOptionsPrefix(ksp_prefix)
        self.kspK.setOperators(self.b.K)
        self.kspK.setType("preonly")
        self.kspK.getPC().setType("lu")
        self.kspK.setFromOptions()

        # Layout-safe workspace vectors
        # omega_full is in the same layout as Mpsi right vector
        self.omega_full = self.b.Mpsi.createVecRight()
        self.tpsi = self.b.Mpsi.createVecLeft()
        self.zpsi = self.b.K.createVecRight()
        self.tmp_free = self.b.A12_row.createVecLeft()

        self.free_idx = self.b.free_idx.astype(np.int32)

    def mult(self, A, x, y):
        # Embed reduced x into omega_full (Dirichlet zeros elsewhere)
        self.omega_full.set(0.0)
        self.omega_full.setValues(self.free_idx, x.getArray(readonly=True), addv=PETSc.InsertMode.INSERT)
        self.omega_full.assemble()

        # tpsi = Mpsi * omega_full
        self.b.Mpsi.mult(self.omega_full, self.tpsi)

        # zpsi = K^{-1} tpsi
        self.zpsi.set(0.0)
        self.kspK.solve(self.tpsi, self.zpsi)
        if int(self.kspK.getConvergedReason()) < 0:
            raise RuntimeError(f"Poisson K solve failed (reason={int(self.kspK.getConvergedReason())}).")

        # y = A11_sub x
        self.b.A11_sub.mult(x, y)

        # tmp_free = A12_row zpsi
        self.tmp_free.set(0.0)
        self.b.A12_row.mult(self.zpsi, self.tmp_free)

        # y += tmp_free
        y.axpy(1.0, self.tmp_free)


class _ShiftInvertShell:
    """
    Manual shift-invert operator:
        T = (A_red - sigma*M)^{-1} * M
    """

    def __init__(self, Ared: PETSc.Mat, M: PETSc.Mat, Pmat: PETSc.Mat,
                 sigma: float, comm, ksp_prefix: str = "ash_"):
        self.comm = comm
        self.Ared = Ared
        self.M = M
        self.Pmat = Pmat
        self.sigma = float(sigma)

        # IMPORTANT: rhs/tmp are outputs of M.mult -> must be in LEFT space of M
        self.rhs = self.M.createVecLeft()
        self.tmp = self.M.createVecLeft()

        self.ksp = PETSc.KSP().create(comm)
        self.ksp.setOptionsPrefix(ksp_prefix)
        # Do NOT call setFromOptions() yet (options can override ksp_type/preonly, etc.)

    def _apply_Ashift(self, x: PETSc.Vec, y: PETSc.Vec):
        # y = (Ared - sigma*M) x
        self.Ared.mult(x, y)
        self.M.mult(x, self.tmp)
        y.axpy(-self.sigma, self.tmp)

    def as_matrix(self) -> PETSc.Mat:
        n = self.M.getSize()[0]

        class _AShiftCtx:
            def __init__(self, outer):
                self.outer = outer
            def mult(self, A, x, y):
                self.outer._apply_Ashift(x, y)

        Ash = PETSc.Mat().create(comm=self.comm)
        Ash.setSizes(((n, None), (n, None)))
        Ash.setType(PETSc.Mat.Type.PYTHON)
        Ash.setPythonContext(_AShiftCtx(self))
        Ash.setUp()
        try:
            Ash.setOption(PETSc.Mat.Option.SYMMETRIC, False)
            Ash.setOption(PETSc.Mat.Option.HERMITIAN, False)
        except Exception:
            pass

        # Attach operator and preconditioner matrix
        self.ksp.setOperators(Ash, self.Pmat)

        # Hard defaults that you actually want (then allow options to tweak)
        self.ksp.setType("fgmres")  # more robust than gmres with strong/variable preconditioning
        self.ksp.setTolerances(rtol=1e-12, atol=0.0, max_it=800)

        pc = self.ksp.getPC()
        pc.setType("lu")  # in serial; if MPI you may want ilu/mumps/etc.

        # Now read user options (optional). If you want *no overrides*, delete this line.
        self.ksp.setFromOptions()

        # Guard against the exact failure mode that gives garbage eigenvalues
        if self.ksp.getType().lower() == "preonly":
            raise RuntimeError(
                "Inner KSP is 'preonly' (likely set via -ash_ksp_type preonly). "
                "That applies only the preconditioner and is NOT (A-sigma M)^{-1} when coupling is nonzero."
            )

        class _TCtx:
            def __init__(self, outer):
                self.outer = outer

            def mult(self, A, x, y):
                # rhs = M x
                self.outer.M.mult(x, self.outer.rhs)

                # solve (A - sigma M) y = rhs
                y.set(0.0)
                self.outer.ksp.solve(self.outer.rhs, y)

                reason = int(self.outer.ksp.getConvergedReason())
                if reason < 0:
                    raise RuntimeError(
                        f"Inner KSP failed: reason={reason}, its={int(self.outer.ksp.getIterationNumber())}"
                    )

                # (highly recommended) sanity check: true relative residual
                # r = (A-sigma M)y - rhs
                r = self.outer.rhs.duplicate()
                self.outer._apply_Ashift(y, r)
                r.axpy(-1.0, self.outer.rhs)
                rel = r.norm() / max(self.outer.rhs.norm(), 1e-30)
                if rel > 1e-7:
                    raise RuntimeError(f"Inner solve too inaccurate: relres={rel:.3e}")

        T = PETSc.Mat().create(comm=self.comm)
        T.setSizes(((n, None), (n, None)))
        T.setType(PETSc.Mat.Type.PYTHON)
        T.setPythonContext(_TCtx(self))
        T.setUp()
        try:
            T.setOption(PETSc.Mat.Option.SYMMETRIC, False)
            T.setOption(PETSc.Mat.Option.HERMITIAN, False)
        except Exception:
            pass

        return T



@dataclass
class Mode:
    lam: complex
    omega_re: fem.Function
    omega_im: fem.Function
    psi_re: fem.Function
    psi_im: fem.Function


class ManifoldEigenSolver:
    """
    Robust solver (Python analogue of your working MATLAB):

      1) Build A_red as MatShell
      2) Build T = (A_red - sigma*M)^{-1} M  as MatShell with inner GMRES(+precond)
      3) EPS on standard eigenproblem T x = mu x (largest magnitude)
      4) Recover lambda = sigma + 1/mu
      5) Reconstruct FEM functions (omega, psi) for each eigenvector
    """

    def __init__(self, mesh, spaces, Omega_star, psi_star, params, facet_tags, outer_tag=1):
        self.mesh = mesh
        self.comm = mesh.comm
        self.Vo = spaces["Vomega"]
        self.Vp = spaces["Vpsi"]
        self.params = params
        self.facet_tags = facet_tags
        self.outer_tag = int(outer_tag)

        self.mu = float(getattr(params, "mu", 0.0))

        # eps = Rdom^2 / R^2
        R = float(getattr(params, "R", 1.0))
        Rdom = float(getattr(params, "Rdom", 1.0))
        self.eps = (Rdom * Rdom) / (R * R)
        print(f"eps in eig solver={self.eps:.3e}")

        self.blocks = assemble_reduced_blocks(
            mesh=self.mesh,
            Vomega=self.Vo,
            Vpsi=self.Vp,
            Omega_star=Omega_star,
            psi_star=psi_star,
            mu=self.mu,
            eps=self.eps,
            facet_tags=self.facet_tags,
            outer_tag=self.outer_tag,
        )

        # Build A_red MatShell once (used by shift-invert)
        self._Ared = self._build_Ared()

        # convenience: reuse Poisson solver from A_red context for reconstruction
        self._ared_ctx: _AredShell = self._Ared.getPythonContext()

    def _build_Ared(self) -> PETSc.Mat:
        ctx = _AredShell(self.blocks, self.comm, ksp_prefix="poisson_")
        n = self.blocks.n_free
        Ared = PETSc.Mat().create(comm=self.comm)
        Ared.setSizes(((n, None), (n, None)))
        Ared.setType(PETSc.Mat.Type.PYTHON)
        Ared.setPythonContext(ctx)
        Ared.setUp()
        try:
            Ared.setOption(PETSc.Mat.Option.SYMMETRIC, False)
            Ared.setOption(PETSc.Mat.Option.HERMITIAN, False)
        except Exception:
            pass
        return Ared

    def solve(
        self,
        k: int = 20,
        sigma: float = -1.0,
        tol: float = 1e-10,
        max_it: int = 400,
        ncv: int | None = None,
        verbose: bool = True,
    ) -> List[Mode]:
        """
        Returns a list of Mode, sorted by increasing Re(lambda).
        """

        comm = self.comm
        n = self.blocks.n_free

        # Preconditioner matrix: P = A11_sub - sigma*M
        Pmat = self.blocks.A11_sub.copy()
        Pmat.axpy(-float(sigma), self.blocks.Moo_sub)
        Pmat.assemble()

        # Build T = (A_red - sigma*M)^{-1} M
        si = _ShiftInvertShell(
            Ared=self._Ared,
            M=self.blocks.Moo_sub,
            Pmat=Pmat,
            sigma=float(sigma),
            comm=comm,
            ksp_prefix="ash_",
        )
        T = si.as_matrix()

        # EPS on standard eigenproblem: T x = mu x
        eps = SLEPc.EPS().create(comm)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setOperators(T)
        eps.setProblemType(SLEPc.EPS.ProblemType.NHEP)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)

        nev = int(k)
        if ncv is None:
            ncv_eff = min(max(8 * nev, 300), min(400, n))
            ncv_eff = max(ncv_eff, nev + 2)
        else:
            ncv_eff = int(min(max(int(ncv), nev + 2), n))

        try:
            eps.setDimensions(nev=nev, ncv=ncv_eff, mpd=max(1, ncv_eff - nev))
        except TypeError:
            eps.setDimensions(nev)

        eps.setTolerances(tol=float(tol), max_it=int(max_it))

        # Prevent user options from switching to JD/LOBPCG etc (those can demand ST)
        eps.setFromOptions()
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_MAGNITUDE)

        if verbose and comm.rank == 0:
            PETSc.Sys.Print(
                f"[PY-EIG] manual shift-invert: n={n}, k={nev}, ncv={ncv_eff}, "
                f"sigma={sigma}, eps={self.eps:.3e}"
            )
            PETSc.Sys.Print(
                "[PY-EIG] inner KSP options prefix: -ash_ksp_*  and preconditioner via -ash_pc_* "
                "(operator uses MatShell, PC uses assembled P=A11-sigma*M)"
            )
            PETSc.Sys.Print(
                "[PY-EIG] poisson K options prefix: -poisson_ksp_*  (default LU)"
            )

        eps.solve()

        nconv = int(eps.getConverged())
        if nconv <= 0:
            raise RuntimeError("No eigenpairs converged for shift-invert operator. Try a different sigma (e.g. -0.5, -2) or tune -ash_ksp_rtol / -ash_pc_type.")

        take = min(nev, nconv)

        # Extract eigenpairs (mu, v), recover lambda
        xr = PETSc.Vec().createMPI(n, comm=comm)
        xi = PETSc.Vec().createMPI(n, comm=comm)

        modes: List[Mode] = []
        for i in range(take):
            mu_val = complex(eps.getEigenpair(i, xr, xi))
            if mu_val == 0:
                continue
            lam = complex(sigma) + 1.0 / mu_val

            # Get reduced eigenvector (real/imag)
            if _is_complex_petsc():
                arr = xr.getArray(readonly=True)
                v_re = np.real(arr).copy()
                v_im = np.imag(arr).copy()
            else:
                v_re = xr.getArray(readonly=True).copy()
                v_im = xi.getArray(readonly=True).copy()

            omega_re, omega_im = self._reconstruct_omega(v_re, v_im)
            psi_re, psi_im = self._reconstruct_psi(v_re, v_im)

            modes.append(Mode(lam=lam, omega_re=omega_re, omega_im=omega_im, psi_re=psi_re, psi_im=psi_im))

        # Sort by increasing real part of lambda (left edge)
        modes.sort(key=lambda m: float(np.real(m.lam)))

        # if verbose and comm.rank == 0:
        #     PETSc.Sys.Print("[PY-EIG] eigenvalues sorted by Re(lambda):")
        #     for j, m in enumerate(modes[:min(len(modes), 20)]):
        #         PETSc.Sys.Print(f"  #{j+1:2d}: lambda = {np.real(m.lam): .15e}  {np.imag(m.lam):+.3e}i")

        return modes

    def _reconstruct_omega(self, v_re: np.ndarray, v_im: np.ndarray) -> Tuple[fem.Function, fem.Function]:
        """Embed reduced eigenvector into full omega Function (Dirichlet zeros on boundary)."""
        omega_re = fem.Function(self.Vo)
        omega_im = fem.Function(self.Vo)
        _vec_data(omega_re)[:] = 0.0
        _vec_data(omega_im)[:] = 0.0

        free = self.blocks.free_idx.astype(np.int64)
        _vec_data(omega_re)[free] = v_re
        _vec_data(omega_im)[free] = v_im
        _ghost_forward(omega_re)
        _ghost_forward(omega_im)
        return omega_re, omega_im
    
    def _reconstruct_psi(self, v_re: np.ndarray, v_im: np.ndarray) -> Tuple[fem.Function, fem.Function]:
        """psi = K^{-1} (Mpsi * omega_full), done separately for Re/Im."""
        ctx = self._ared_ctx  # reuse Poisson KSP + layouts

        def solve_one(v_part: np.ndarray) -> fem.Function:
            # omega_full
            ctx.omega_full.set(0.0)
            ctx.omega_full.setValues(self.blocks.free_idx, v_part, addv=PETSc.InsertMode.INSERT)
            ctx.omega_full.assemble()

            # tpsi = Mpsi * omega_full
            self.blocks.Mpsi.mult(ctx.omega_full, ctx.tpsi)

            # zpsi = K^{-1} tpsi
            ctx.zpsi.set(0.0)
            ctx.kspK.solve(ctx.tpsi, ctx.zpsi)
            if int(ctx.kspK.getConvergedReason()) < 0:
                raise RuntimeError("Poisson solve failed during reconstruction.")

            psi = fem.Function(self.Vp)
            _vec_data(psi)[:] = ctx.zpsi.getArray(readonly=True)
            _ghost_forward(psi)
            return psi

        psi_re = solve_one(v_re)
        psi_im = solve_one(v_im)
        return psi_re, psi_im
    
    # Export matrices to MATLAB
    # Add this method inside ManifoldEigenSolver
    def export_to_matlab(self, path: str):
        """
        Export blocks to a single .mat that MATLAB can use to reconstruct:

            A_red*x = A11_sub*x + A12_row * ( K \ ( Mpsi * (P*x) ) )
            generalized EVP: A_red*x = lambda * Moo_sub*x

        Exports (MATLAB names):
        A11_sub, A12_row, Moo_sub, Mpsi, K : MATLAB sparse
        free_idx : (n_free,1)  1-based GLOBAL ω indices
        n_o, n_p : total dof counts (GLOBAL)
        """
        import numpy as np

        comm = self.mesh.comm

        # ---- grab matrices from either self.mats dict or self.blocks dataclass ----
        if hasattr(self, "mats"):          # dict-style
            A11_sub = self.mats["A11_sub"]
            A12_row = self.mats["A12_row"]
            Moo_sub = self.mats["Moo_sub"]
            Mpsi    = self.mats["Mpsi"]
            K       = self.mats["K"]
            free_idx_local = self.mats["free_idx"]  # 0-based global
            # Prefer global dof counts if available; else compute from spaces
            n_o = int(getattr(self.mats.get("n_o", None), "__int__", lambda: self.Vo.dofmap.index_map.size_global)())
            n_p = int(getattr(self.mats.get("n_p", None), "__int__", lambda: self.Vp.dofmap.index_map.size_global)())
        else:                               # dataclass-style
            b = self.blocks
            A11_sub = b.A11_sub
            A12_row = b.A12_row
            Moo_sub = b.Moo_sub
            Mpsi    = b.Mpsi
            K       = b.K
            free_idx_local = b.free_idx      # 0-based global
            n_o = int(getattr(b, "n_o", self.Vo.dofmap.index_map.size_global))
            n_p = int(getattr(b, "n_p", self.Vp.dofmap.index_map.size_global))

        # ---- gather free_idx and all matrices to rank 0 as SciPy CSR ----
        free_idx_global = _gather_free_idx_global(free_idx_local, comm)

        A11_csr = _petsc_mat_to_scipy_csr_global(A11_sub, comm)
        A12_csr = _petsc_mat_to_scipy_csr_global(A12_row, comm)
        Moo_csr = _petsc_mat_to_scipy_csr_global(Moo_sub, comm)
        Mpsi_csr = _petsc_mat_to_scipy_csr_global(Mpsi, comm)
        K_csr    = _petsc_mat_to_scipy_csr_global(K, comm)

        if comm.rank != 0:
            return  # only rank 0 writes

        out = {
            "A11_sub": A11_csr,
            "A12_row": A12_csr,
            "Moo_sub": Moo_csr,
            "Mpsi":    Mpsi_csr,
            "K":       K_csr,
            # MATLAB is 1-based:
            "free_idx": (free_idx_global.astype(np.int64) + 1).reshape(-1, 1),
            "n_o": np.array([int(n_o)], dtype=np.int64),
            "n_p": np.array([int(n_p)], dtype=np.int64),
        }

        _save_matlab_sparse(path, out, comm)
        print(f"[EXPORT] wrote MATLAB blocks to: {path}")


# Export the Matrices to MATLAB
def _petsc_mat_to_scipy_csr_global(A_petsc, comm):
    """
    Gather a (possibly MPI) PETSc Mat to rank-0 as a SciPy CSR matrix.
    Returns:
      - csr on rank 0
      - None on other ranks
    """
    import numpy as np
    import scipy.sparse as sp

    A = A_petsc.convert("aij")  # ensure AIJ/CSR storage
    (m, n) = A.getSize()
    rstart, rend = A.getOwnershipRange()

    ia, ja, a = A.getValuesCSR()  # CSR of local rows
    ia = np.asarray(ia, dtype=np.int64)
    ja = np.asarray(ja, dtype=np.int64)
    a = np.asarray(a)  # float64 or complex128

    # Convert local CSR -> local COO with GLOBAL row indices
    nloc = rend - rstart
    if nloc != (len(ia) - 1):
        # Very defensive; should not happen for AIJ
        raise RuntimeError("CSR row pointer length mismatch in PETSc Mat.")

    counts = ia[1:] - ia[:-1]
    rows = np.repeat(np.arange(rstart, rend, dtype=np.int64), counts)
    cols = ja.astype(np.int64, copy=False)
    data = a

    if comm.size == 1:
        return sp.csr_matrix((data, (rows, cols)), shape=(m, n))

    gathered = comm.gather((rows, cols, data), root=0)
    if comm.rank != 0:
        return None

    rows_all = np.concatenate([g[0] for g in gathered]) if gathered else np.empty((0,), dtype=np.int64)
    cols_all = np.concatenate([g[1] for g in gathered]) if gathered else np.empty((0,), dtype=np.int64)
    data_all = np.concatenate([g[2] for g in gathered]) if gathered else np.empty((0,), dtype=data.dtype)

    return sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(m, n))


def _gather_free_idx_global(free_idx_local, comm):
    """
    Gather per-rank free_idx (GLOBAL ω dof ids, 0-based) into a single
    rank-0 array in the correct global order.
    """
    import numpy as np
    free_idx_local = np.asarray(free_idx_local, dtype=np.int64)
    if comm.size == 1:
        return free_idx_local

    gathered = comm.gather(free_idx_local, root=0)
    if comm.rank != 0:
        return None

    free_idx = np.concatenate(gathered) if gathered else np.empty((0,), dtype=np.int64)
    # Ensure a deterministic increasing global ordering (matches typical IS ordering)
    free_idx.sort()
    return free_idx


def _save_matlab_sparse(path, outdict, comm):
    """
    Save on rank 0 only. SciPy sparse matrices become MATLAB sparse.
    """
    import scipy.io as sio
    if comm.rank == 0:
        sio.savemat(path, outdict, do_compression=True)


## Plotting functions for eigenmodes
import os
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def save_modes_plots_real(
    mesh,
    modes,
    *,
    outdir="eig_plots",
    prefix="eig",
    n_plot=None,
    plot_stream=False,
    stream_levels=40,
    stream_linewidth=0.8,
):
    """
    Save plots for REAL parts of eigenmodes:
      - vorticity heatmap of mode.omega_re (always)
      - optionally streamfunction contours of mode.psi_re (if plot_stream=True and psi_re exists)

    Output folder is created INSIDE the folder that contains THIS python file.
    """
    # Put output folder inside the code folder (directory of this file)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    outdir_path = outdir if os.path.isabs(outdir) else os.path.join(base_dir, outdir)
    os.makedirs(outdir_path, exist_ok=True)

    if n_plot is None:
        n_plot = len(modes)
    n_plot = min(int(n_plot), len(modes))

    # ----- Vorticity triangulation (CG1 nodal values align with vertices) -----
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    conn = mesh.topology.connectivity(tdim, 0)
    cells = conn.array.reshape(-1, 3)  # triangles

    X = mesh.geometry.x
    xcoord = X[:, 0]
    ycoord = X[:, 1]
    tri_vtx = mtri.Triangulation(xcoord, ycoord, triangles=cells)

    def _func_array(u):
        try:
            return u.vector.array
        except AttributeError:
            return u.x.array

    # ----- Optional: prepare streamfunction triangulation (cell_dofs on CG1) -----
    tri_dofs = None
    xy_dofs = None
    if plot_stream:
        # We will build these lazily from the first available psi_re
        pass

    for j in range(n_plot):
        m = modes[j]
        lam = complex(m.lam)

        # ========= Vorticity heatmap (Re part only) =========
        Omega = m.omega_re
        vals = _func_array(Omega)
        vfile = os.path.join(outdir_path, f"{prefix}_{j+1:03d}_OmegaRe.png")

        plt.figure()
        cf = plt.tricontourf(tri_vtx, vals, levels=64)
        plt.colorbar(cf, label=r"$\Re(\delta\Omega)$")
        plt.gca().set_aspect("equal", "box")
        plt.title(rf"$\lambda = {lam.real:.6e} {lam.imag:+.6e}i$")
        plt.tight_layout()
        plt.savefig(vfile, dpi=200)
        plt.close()

        # ========= Streamlines (optional, Re part only) =========
        if plot_stream and hasattr(m, "psi_re") and (m.psi_re is not None):
            psi = m.psi_re
            Vpsi = psi.function_space

            # Build triangulation in dof-layout once
            if tri_dofs is None:
                raw = Vpsi.tabulate_dof_coordinates()
                if raw.ndim == 1:
                    gdim = mesh.geometry.x.shape[1]  # 2 or 3
                    xy_dofs = raw.reshape(-1, gdim)[:, :2]
                else:
                    xy_dofs = raw[:, :2]

                num_cells = mesh.topology.index_map(tdim).size_local
                tri_dofs = np.array([Vpsi.dofmap.cell_dofs(c) for c in range(num_cells)], dtype=np.int32)

            tri_psi = mtri.Triangulation(xy_dofs[:, 0], xy_dofs[:, 1], triangles=tri_dofs)
            psivals = _func_array(psi)

            sfile = os.path.join(outdir_path, f"{prefix}_{j+1:03d}_PsiRe_stream.png")

            plt.figure(figsize=(6, 6), dpi=160)
            plt.tricontourf(tri_psi, psivals, levels=stream_levels, alpha=0.75)
            plt.tricontour(tri_psi, psivals, levels=stream_levels, colors="k",
                           linewidths=stream_linewidth, alpha=0.9)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.axis("off")
            plt.title(rf"$\lambda = {lam.real:.6e} {lam.imag:+.6e}i$")
            plt.tight_layout()
            plt.savefig(sfile, bbox_inches="tight", pad_inches=0, dpi=200)
            plt.close()



