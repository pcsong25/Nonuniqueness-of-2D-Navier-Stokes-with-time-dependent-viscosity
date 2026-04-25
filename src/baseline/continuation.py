# --- file: continuation.py ---
from ast import If
import numpy as np
import ufl
from petsc4py import PETSc
from dolfinx import fem


class ContinuationSolver:
    """
    σ–continuation + Newton on the manifold { (Ω, ψ): Δψ = Ω }.

    For each σ > 0:
      (1) Impose Ω|∂B = σ R^{-1/μ} \barΩ(θ) once.
      (2) Solve Poisson once to get ψ with ∂nψ = b_σ (initializes the flux).
      (3) Newton iterations:
            - assemble with L2 ≡ 0 (constraint row K δψ − M δΩ = 0),
            - solve A δx = b,
            - update Ω ← Ω + δΩ (δΩ|∂B = 0 preserves BC),
            - update ψ ← ψ + δψ (∂nδψ = 0 preserves flux b_σ),
            - stop on relative Ω-step.
      (4) Done (no extra Poisson solve needed).
    """

    def __init__(self, assembler, poisson, params,
                 facet_tags=None, outer_tag=1, fourier_bc=None):

        self.assembler = assembler
        self.poisson = poisson
        self.params = params

        # Boundary tagging/BC providers
        self.facet_tags = facet_tags if facet_tags is not None \
            else getattr(assembler, "facet_tags", None) \
            or getattr(poisson, "facet_tags", None)

        self.outer_tag = int(outer_tag) if outer_tag is not None \
            else getattr(assembler, "outer_tag", 1)

        self.fourier_bc = fourier_bc if fourier_bc is not None \
            else getattr(poisson, "fourier_bc", None)

        # Common scalars
        self.mu = float(getattr(params, "mu", 2.0))
        self.R = float(getattr(params, "R", 100.0))

    # ---------- utilities ----------
    @staticmethod
    def _vec_data(func):
        try:
            return func.vector.array
        except AttributeError:
            return func.x.array

    @staticmethod
    def _ghost_forward(func):
        try:
            func.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                    mode=PETSc.ScatterMode.FORWARD)
        except AttributeError:
            if hasattr(func.x, "scatter_forward"):
                func.x.scatter_forward()

    @staticmethod
    def _owned_size(V):
        im = V.dofmap.index_map
        bs = getattr(im, "block_size", None)
        if bs is None:
            bs = getattr(im, "bs", None)
        if bs is None:
            bs = getattr(V.dofmap, "index_map_bs", 1)
        return im.size_local * int(bs)

    @classmethod
    def _split_block_vec(cls, x: PETSc.Vec, Vomega, Vpsi):
        n_o = cls._owned_size(Vomega)
        n_p = cls._owned_size(Vpsi)
        is_o = PETSc.IS().createStride(n_o, 0, 1)
        is_p = PETSc.IS().createStride(n_p, n_o, 1)
        x1 = x.getSubVector(is_o)
        x2 = x.getSubVector(is_p)
        return (x1, is_o), (x2, is_p)

    @staticmethod
    def _vec_local_array(v: PETSc.Vec) -> np.ndarray:
        try:
            return v.array
        except AttributeError:
            with v.localForm() as lf:
                return lf.array.copy()

    @staticmethod
    def _dof_xy(V):
        raw = V.tabulate_dof_coordinates()
        if raw.ndim == 1:
            gdim = 3 if raw.size % 3 == 0 else 2
            xy = raw.reshape(-1, gdim)[:, :2]
        else:
            xy = raw[:, :2]
        return xy

    # ---------- Ω boundary preset ----------
    def _eval_Omega_bar(self, theta: np.ndarray) -> np.ndarray:
        if self.fourier_bc is not None and hasattr(self.fourier_bc, "Omega_bar"):
            return np.asarray(self.fourier_bc.Omega_bar(theta), dtype=float)
        if callable(self.fourier_bc):
            return np.asarray(self.fourier_bc(theta), dtype=float)
        coeffs = getattr(self.params, "omega_bar_coeffs", None)
        if coeffs is not None:
            a0, a_cos, b_sin = coeffs
            a_cos = np.asarray(a_cos, dtype=float)
            b_sin = np.asarray(b_sin, dtype=float)
            th = np.asarray(theta, dtype=float)
            res = np.full_like(th, float(a0), dtype=float)
            K = len(a_cos)
            for k in range(1, K + 1):
                res += a_cos[k - 1] * np.cos(k * th) + b_sin[k - 1] * np.sin(k * th)
            return res
        raise RuntimeError("Provide fourier_bc.Omega_bar(θ) or params.omega_bar_coeffs.")

    def _apply_omega_dirichlet_on_boundary(self, Omega, Vomega, sigma: float):
        if self.facet_tags is None:
            return
        mesh = Vomega.mesh
        bfacets = self.facet_tags.find(int(self.outer_tag))
        bdofs = fem.locate_dofs_topological(Vomega, mesh.topology.dim - 1, bfacets)
        if len(bdofs) == 0:
            return
        raw = Vomega.tabulate_dof_coordinates()
        if raw.ndim == 1:
            gdim = 3 if raw.size % 3 == 0 else 2
            xy = raw.reshape(-1, gdim)[:, :2][bdofs]
        else:
            xy = raw[:, :2][bdofs]
        theta = np.arctan2(xy[:, 1], xy[:, 0])
        g = (self.R ** (-1.0 / self.mu)) * self._eval_Omega_bar(theta) * float(sigma)
        arr = self._vec_data(Omega)
        arr[bdofs.astype(np.int64)] = g
        self._ghost_forward(Omega)

    # ---------- main driver ----------
    def run(self, mesh, spaces):
        Vomega, Vpsi = spaces["Vomega"], spaces["Vpsi"]

        # Initial guess (Ω, ψ) = (0, 0)
        Omega = fem.Function(Vomega); self._vec_data(Omega)[:] = 0.0; self._ghost_forward(Omega)
        psi   = fem.Function(Vpsi);   self._vec_data(psi)[:]   = 0.0; self._ghost_forward(psi)

        # Skip σ = 0; start from the first positive σ
        sigmas = [float(s) for s in self.params.sigma_grid if float(s) > 0.0]
        sigma_prev = sigmas[0]
        First_step = True

        if not sigmas:
            raise ValueError("sigma_grid must contain a positive σ to start continuation.")

        it_sigma = 0
        for sigma in sigmas:
            it_sigma += 1
            if First_step:
                self._apply_omega_dirichlet_on_boundary(Omega, Vomega, sigma)
                First_step = False
            else:
                ratio = float(sigma) / float(sigma_prev)
                # Scale entire Ω to match the boundary data to get initial guess
                self._vec_data(Omega)[:] *= ratio
                # # Only scale the boundary values
                # self._apply_omega_dirichlet_on_boundary(Omega, Vomega, sigma)
                self._ghost_forward(Omega)
                sigma_prev = sigma
            # (1) Impose Ω Dirichlet once for this σ
            # self._apply_omega_dirichlet_on_boundary(Omega, Vomega, sigma)
  
            # (2) Initialize ψ for this σ to have the correct Neumann flux b_σ
            psi = self.poisson.solve_state(Omega, sigma=float(sigma))

            # (3) Newton iterations
            for it in range(self.params.max_newton):
                A, b = self.assembler.assemble(Omega, psi, mu=self.mu)

                ksp = PETSc.KSP().create(mesh.comm)
                ksp.setOperators(A)
                ksp.setType("preonly"); ksp.getPC().setType("lu"); ksp.setFromOptions()
                x = b.duplicate(); x.set(0.0)
                ksp.solve(b, x)

                # Split updates
                (x1, is1), (x2, is2) = self._split_block_vec(x, Vomega, Vpsi)

                # Ω ← Ω + δΩ  (δΩ|∂B=0 preserves BC set once per σ)
                try:
                    Omega.vector.axpy(1.0, x1); self._ghost_forward(Omega)
                except AttributeError:
                    d1 = self._vec_local_array(x1)
                    arr = self._vec_data(Omega); arr[:d1.size] += d1; self._ghost_forward(Omega)

                # ψ ← ψ + δψ  (∂nδψ=0 preserves flux b_σ)
                try:
                    psi.vector.axpy(1.0, x2); self._ghost_forward(psi)
                except AttributeError:
                    d2 = self._vec_local_array(x2)
                    arrp = self._vec_data(psi); arrp[:d2.size] += d2; self._ghost_forward(psi)

                # Convergence: relative Ω-step
                d1norm = np.linalg.norm(self._vec_local_array(x1))
                onorm  = np.linalg.norm(self._vec_data(Omega)) + 1e-16
                rel = d1norm / onorm

                x.restoreSubVector(is1, x1)
                x.restoreSubVector(is2, x2)

                if rel < self.params.tol_newton:
                    break
            # print the relative error for this sigma; print for every 5 iterations of sigma
            if it_sigma % 5 == 0:
                print(f"[sigma={sigma:.4e}] Newton converged in {it+1} iters, rel = {rel:.3e}")


            # (4) No extra Poisson solve needed here (we stayed on Δψ = Ω with correct flux)

        return Omega, psi


