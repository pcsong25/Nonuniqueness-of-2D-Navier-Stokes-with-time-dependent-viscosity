# --- file: newton_direct.py ---
import numpy as np
from dolfinx import fem
from petsc4py import PETSc


class DirectNewtonSolver:
    """
    Monolithic Newton solver that STARTS FROM AN EXTERNAL INITIAL GUESS:
       use run_from(mesh, spaces, Omega0, psi0).

    - No analytic guess is constructed here.
    - δΩ satisfies homogeneous Dirichlet and δψ homogeneous Neumann (with one pinned dof),
      so boundary data are preserved by the updates and are never reimposed inside Newton.
    - Optional Armijo backtracking (linesearch=True) to stabilize coarse/advective cases.

    Requirements
    ------------
    assembler: object with method assemble(Omega, psi, mu) -> (A, b)
               (A is PETSc.Mat AIJ, b is PETSc.Vec for the monolithic 2×2 block)
    poisson  : kept only for interface symmetry (not used here)
    params   : has at least fields: mu, tol_newton, max_newton
    """

    def __init__(self, assembler, poisson, params,
                 linesearch: bool = False, ls_max_steps: int = 6, ls_shrink: float = 0.5,
                 verbose: bool = True):
        self.assembler = assembler
        self.poisson = poisson
        self.params = params

        self.mu = float(getattr(params, "mu", 2.0))
        self.tol = float(getattr(params, "tol_newton", 1e-10))
        self.maxit = int(getattr(params, "max_newton", 12))

        self.linesearch = bool(linesearch)
        self.ls_max_steps = int(ls_max_steps)
        self.ls_shrink = float(ls_shrink)
        self.verbose = bool(verbose)

    # -------- helpers: version-robust vector ops --------
    @staticmethod
    def _vec_data(fn: fem.Function):
        try:
            return fn.vector.array
        except AttributeError:
            return fn.x.array

    @staticmethod
    def _ghost_fwd(fn: fem.Function):
        try:
            fn.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            if hasattr(fn.x, "scatter_forward"):
                fn.x.scatter_forward()

    @staticmethod
    def _owned_size(V):
        im = V.dofmap.index_map
        bs = getattr(im, "block_size", None) or getattr(im, "bs", None) or getattr(V.dofmap, "index_map_bs", 1)
        return im.size_local * int(bs)

    @classmethod
    def _split_block_vec(cls, x: PETSc.Vec, Vomega, Vpsi):
        """Return ((xΩ, ISΩ), (xψ, ISψ)) as PETSc subvectors."""
        n_o = cls._owned_size(Vomega)
        n_p = cls._owned_size(Vpsi)
        is_o = PETSc.IS().createStride(n_o, 0, 1)
        is_p = PETSc.IS().createStride(n_p, n_o, 1)
        x1 = x.getSubVector(is_o)
        x2 = x.getSubVector(is_p)
        return (x1, is_o), (x2, is_p)

    @staticmethod
    def _subvec_array(v: PETSc.Vec) -> np.ndarray:
        try:
            return v.array
        except AttributeError:
            with v.localForm() as lf:
                return lf.array.copy()

    def _apply_update(self, Omega, Vomega, psi, Vpsi, x_sub_pair, alpha: float):
        """Add alpha * (δΩ, δψ) into (Ω, ψ) in-place."""
        (x1, is1), (x2, is2) = x_sub_pair

        # Ω update
        try:
            Omega.vector.axpy(alpha, x1)
            self._ghost_fwd(Omega)
        except AttributeError:
            d1 = self._subvec_array(x1)
            arr = self._vec_data(Omega)
            arr[:d1.size] += alpha * d1
            self._ghost_fwd(Omega)

        # ψ update
        try:
            psi.vector.axpy(alpha, x2)
            self._ghost_fwd(psi)
        except AttributeError:
            d2 = self._subvec_array(x2)
            arrp = self._vec_data(psi)
            arrp[:d2.size] += alpha * d2
            self._ghost_fwd(psi)

    def _save_state(self, Omega, psi):
        return self._vec_data(Omega).copy(), self._vec_data(psi).copy()

    def _restore_state(self, Omega, psi, buf_O, buf_P):
        self._vec_data(Omega)[:] = buf_O
        self._vec_data(psi)[:] = buf_P
        self._ghost_fwd(Omega); self._ghost_fwd(psi)

    # -------- main APIs --------
    def run(self, mesh, spaces):
        """Not supported anymore: we require an external initial guess."""
        raise RuntimeError("DirectNewtonSolver.run(): please call run_from(...) with (Omega0, psi0).")

    def run_from(self, mesh, spaces, Omega0: fem.Function, psi0: fem.Function):
        """
        Start Newton from the provided (Omega0, psi0). Returns the converged (Omega, psi).
        """
        Vomega, Vpsi = spaces["Vomega"], spaces["Vpsi"]

        # Copy initial guess into local working functions (so caller can reuse their copies)
        Omega = fem.Function(Vomega); self._vec_data(Omega)[:] = self._vec_data(Omega0)
        psi   = fem.Function(Vpsi);   self._vec_data(psi)[:]   = self._vec_data(psi0)
        self._ghost_fwd(Omega); self._ghost_fwd(psi)

        # Initial residual norm (b-norm) for reporting
        A, b = self.assembler.assemble(Omega, psi, mu=self.mu)
        bnorm = b.norm(PETSc.NormType.NORM_2)

        if self.verbose:
            print(f"[Newton] it=0  ||b||={bnorm:.3e}")

        for it in range(1, self.maxit + 1):
            # Linear solve A x = b
            ksp = PETSc.KSP().create(mesh.comm)
            ksp.setOperators(A)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.setFromOptions()

            x = b.duplicate(); x.set(0.0)
            ksp.solve(b, x)

            # Split update
            x_pair = self._split_block_vec(x, Vomega, Vpsi)

            # Step size (Armijo backtracking if enabled)
            alpha = 1.0
            if self.linesearch:
                buf_O, buf_P = self._save_state(Omega, psi)
                for _ in range(self.ls_max_steps):
                    # Trial update
                    self._restore_state(Omega, psi, buf_O, buf_P)
                    self._apply_update(Omega, Vomega, psi, Vpsi, x_pair, alpha)
                    # Evaluate new residual
                    A_trial, b_trial = self.assembler.assemble(Omega, psi, mu=self.mu)
                    bnorm_new = b_trial.norm(PETSc.NormType.NORM_2)
                    if bnorm_new < bnorm:      # simple monotone decrease
                        A, b = A_trial, b_trial
                        bnorm = bnorm_new
                        break
                    alpha *= self.ls_shrink
                else:
                    # If no decrease achieved, keep the smallest step anyway
                    self._restore_state(Omega, psi, buf_O, buf_P)
                    self._apply_update(Omega, Vomega, psi, Vpsi, x_pair, alpha)
                    A, b = self.assembler.assemble(Omega, psi, mu=self.mu)
                    bnorm = b.norm(PETSc.NormType.NORM_2)
            else:
                # Full step
                self._apply_update(Omega, Vomega, psi, Vpsi, x_pair, alpha=1.0)
                A, b = self.assembler.assemble(Omega, psi, mu=self.mu)
                bnorm = b.norm(PETSc.NormType.NORM_2)

            # Step-size/relative Ω update criterion
            d1 = self._subvec_array(x_pair[0][0])
            onorm = np.linalg.norm(self._vec_data(Omega)) + 1e-16
            rel = np.linalg.norm(d1) / onorm

            # Restore PETSc subvectors
            x.restoreSubVector(x_pair[0][1], x_pair[0][0])
            x.restoreSubVector(x_pair[1][1], x_pair[1][0])

            if self.verbose:
                print(f"[Newton] it={it}  ||b||={bnorm:.3e}  relΩ={rel:.3e}  alpha={alpha:.2f}")

            if rel < self.tol:
                if self.verbose:
                    print("[Newton] Converged by relative Ω-step.")
                break

        return Omega, psi

