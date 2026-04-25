# --- file: eigs_linearized.py ---
import numpy as np
import ufl
from petsc4py import PETSc
try:
    from slepc4py import SLEPc
except Exception as e:
    raise RuntimeError("slepc4py is required. Install it in your fenicsx environment.") from e

from dolfinx import fem

# Version-robust assembly imports
try:
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
except Exception:
    from dolfinx.fem import assemble_matrix, assemble_vector


# -------------------------
# Helpers for global index sets
# -------------------------
def _owned_global_dofs(V):
    """Return owned (global) dof ids for scalar CG1 space V."""
    im = V.dofmap.index_map
    start, end = im.local_range
    return np.arange(start, end, dtype=np.int32)


def _to_global_dofs(V, local_dofs):
    """Map local (owned) dof ids -> global ids using the start offset."""
    start, _ = V.dofmap.index_map.local_range
    return start + local_dofs.astype(np.int64)


# -------------------------
# Assembly of linearized forms (pure Galerkin, NO SUPG)
# -------------------------
def assemble_linearized_forms(
    mesh,
    Vomega,
    Vpsi,
    Omega_star: fem.Function,
    psi_star: fem.Function,
    mu: float,
    eps: float,
    facet_tags,
    outer_tag: int = 1,
):
    """
    Build matrices:
      A11 : (+eps Δ − v*·∇ − 1) on Vomega  (full)
      A12 : (∇^⊥ φ)·∇Ω* tested in Vomega (omega-rows × psi-cols)
      Moo : mass on Vomega (full)
      Mpsi: mass on Vpsi   (full)
      K   : Neumann stiffness on Vpsi with one pinned dof (gauge)

    And restriction data:
      is_free : PETSc.IS of free (non-Dirichlet) dofs for δω in Vomega (GLOBAL ids)
      is_all_psi : PETSc.IS of all ψ dofs (GLOBAL ids)
      free_idx   : GLOBAL ids of free ω dofs
    """
    # Trial / test
    w = ufl.TestFunction(Vomega)
    u = ufl.TrialFunction(Vomega)
    phi_t = ufl.TestFunction(Vpsi)
    phi = ufl.TrialFunction(Vpsi)

    # Frozen coefficients
    x = ufl.SpatialCoordinate(mesh)
    gpsi = ufl.grad(psi_star)
    vstar = ufl.as_vector((-gpsi[1], gpsi[0])) - mu * ufl.as_vector((x[0], x[1]))
    grad_Os = ufl.grad(Omega_star)

    # Forms
    a11_form = (
        eps * ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
        + ufl.inner(ufl.grad(u), vstar) * w * ufl.dx
        - ufl.inner(u, w) * ufl.dx
    )
    gphi = ufl.grad(phi)
    rot_phi = ufl.as_vector((-gphi[1], gphi[0]))
    a12_form = ufl.inner(rot_phi, grad_Os) * w * ufl.dx

    mass_omega_form = u * w * ufl.dx
    mass_psi_form = phi * phi_t * ufl.dx
    K_form = ufl.inner(-ufl.grad(phi), ufl.grad(phi_t)) * ufl.dx  # Neumann natural BC

    # Assemble full matrices
    A11 = assemble_matrix(fem.form(a11_form)); A11.assemble()
    A12 = assemble_matrix(fem.form(a12_form)); A12.assemble()
    Moo = assemble_matrix(fem.form(mass_omega_form)); Moo.assemble()
    Mpsi = assemble_matrix(fem.form(mass_psi_form)); Mpsi.assemble()
    K = assemble_matrix(fem.form(K_form)); K.assemble()

    # Dirichlet dofs for δω on outer boundary (GLOBAL ids)
    outer_facets = facet_tags.find(int(outer_tag))
    bdofs_o_local = fem.locate_dofs_topological(Vomega, mesh.topology.dim - 1, outer_facets)
    bdofs_o_glob = _to_global_dofs(Vomega, bdofs_o_local)

    all_o_glob = _owned_global_dofs(Vomega)  # owned global ids
    mask = np.ones(all_o_glob.size, dtype=bool)
    if bdofs_o_glob.size > 0:
        bset = set(int(i) for i in bdofs_o_glob)
        for j, gid in enumerate(all_o_glob):
            if gid in bset:
                mask[j] = False
    free_glob = all_o_glob[mask].astype(np.int32)
    is_free = PETSc.IS().createGeneral(free_glob, comm=mesh.comm)

    # ψ pin (gauge): pick a boundary ψ dof if available (GLOBAL id), else first owned
    bdofs_p_local = fem.locate_dofs_topological(Vpsi, mesh.topology.dim - 1, outer_facets)
    if bdofs_p_local.size > 0:
        pin_dof = int(_to_global_dofs(Vpsi, bdofs_p_local[:1])[0])
    else:
        pin_dof = int(_owned_global_dofs(Vpsi)[0])

    # Apply pin to K: zero row+col and set diagonal to 1.0
    K.zeroRowsColumns([pin_dof], 1.0)

    # IS for all ψ owned dofs (GLOBAL ids)
    all_p_glob = _owned_global_dofs(Vpsi)
    is_all_psi = PETSc.IS().createGeneral(all_p_glob, comm=mesh.comm)

    # Restricted submatrices (createSubMatrix is portable)
    A11_sub = A11.createSubMatrix(is_free, is_free)
    Moo_sub = Moo.createSubMatrix(is_free, is_free)
    A12_row = A12.createSubMatrix(is_free, is_all_psi)

    # Sizes (owned)
    n_o_owned = all_o_glob.size
    n_p_owned = all_p_glob.size

    return dict(
        A11_sub=A11_sub,
        A12_row=A12_row,
        Moo_sub=Moo_sub,
        Mpsi=Mpsi,
        K=K,
        is_free=is_free,
        free_idx=free_glob,      # GLOBAL ids of free ω dofs
        is_all_psi=is_all_psi,
        pin_dof=pin_dof,
        n_o=n_o_owned,
        n_p=n_p_owned,
    )


# -------------------------
# MatShell context for A_red x = A11_sub x + A12_row * (K^{-1} * (Mpsi * P x))
# -------------------------
class _AredShell:
    def __init__(self, mats, comm):
        self.comm = comm
        self.A11 = mats["A11_sub"]
        self.A12_row = mats["A12_row"]
        self.Mpsi = mats["Mpsi"]
        self.K = mats["K"]

        self.n_free = mats["Moo_sub"].getSize()[0]
        self.n_psi = mats["n_p"]
        self.n_o = mats["n_o"]

        # KSP for K (Neumann + gauge → SPD; LU is robust)
        self.kspK = PETSc.KSP().create(comm)
        self.kspK.setOperators(self.K)
        self.kspK.setType("preonly")
        pc = self.kspK.getPC(); pc.setType("lu")
        self.kspK.setFromOptions()

        # Workspace vectors
        self.tmp_full_omega = PETSc.Vec().createMPI(self.n_o, comm=comm)   # ω full
        self.t_psi = PETSc.Vec().createMPI(self.n_psi, comm=comm)          # RHS in ψ
        self.z_psi = PETSc.Vec().createMPI(self.n_psi, comm=comm)          # K^{-1} RHS
        self.tmp_free = PETSc.Vec().createMPI(self.n_free, comm=comm)      # free ω
        self.free_idx = mats["free_idx"].astype(np.int32)                   # GLOBAL free ids

    def mult(self, A, x, y):
        """
        y = A_red * x = A11_sub x + A12_row * (K^{-1} * (Mpsi * P x))
        P embeds ω-free vector into ω-full by putting zeros on boundary dofs.
        """
        # Embed x (ω-free) -> tmp_full_omega (ω-full)
        self.tmp_full_omega.set(0.0)
        self.tmp_full_omega.setValues(self.free_idx, x.getArray(readonly=True), addv=PETSc.InsertMode.INSERT)
        self.tmp_full_omega.assemble()

        # t_psi = Mpsi * tmp_full_omega
        self.Mpsi.mult(self.tmp_full_omega, self.t_psi)

        # Solve K z = t_psi
        self.z_psi.set(0.0)
        self.kspK.solve(self.t_psi, self.z_psi)

        # y = A11_sub * x
        self.A11.mult(x, y)

        # tmp_free = A12_row * z_psi
        self.tmp_free.set(0.0)
        self.A12_row.mult(self.z_psi, self.tmp_free)

        # y += tmp_free
        y.axpy(1.0, self.tmp_free)


# -------------------------
# Public solver class
# -------------------------
class ManifoldEigenSolver:
    """
    Solve A_red x = λ M x (reduced manifold), find k eigenpairs with the
    SMALLEST real part (left edge). Robust variant uses ST=PRECOND with a
    sparse shifted preconditioner P = A11_sub - σ Moo_sub (no shift set on ST).
    """

    def __init__(self, mesh, spaces, Omega_star, psi_star, params, facet_tags, outer_tag=1):
        self.mesh = mesh
        self.Vo = spaces["Vomega"]; self.Vp = spaces["Vpsi"]
        self.params = params
        self.mu = float(params.mu)
        # epsilon from params
        eps = (params.Rdom**2) / (params.R**2)
        self.eps = float(eps)
        print(f"eps in eig solver = {self.eps}")

        self.mats = assemble_linearized_forms(
            mesh, self.Vo, self.Vp, Omega_star, psi_star, self.mu, self.eps, facet_tags, outer_tag
        )

    def solve(self, k=3, max_it=2000, tol=1e-8,
          which="TARGET_REAL", target_real=-1.0,
          ncv=None, mpd=None, verbose=True):
        """
        Safe & robust left-edge eigen solve:
        - EPS = KRYLOVSCHUR (no ST)
        - generalized problem A_red x = λ M x
        - harmonic extraction + TARGET_REAL selection
        - two-sided DISABLED (MatShell has no multTranspose)
        """
        from slepc4py import SLEPc
        from petsc4py import PETSc
        import numpy as np
        from dolfinx import fem

        comm = self.mesh.comm
        Moo_sub = self.mats["Moo_sub"]

        # --- Reduced operator as MatShell (A_red) ---
        ared_ctx = _AredShell(self.mats, comm)
        Ared = PETSc.Mat().create(comm=comm)
        n = ared_ctx.n_free
        Ared.setSizes(((n, None), (n, None)))
        Ared.setType(PETSc.Mat.Type.PYTHON)
        Ared.setPythonContext(ared_ctx)
        Ared.setUp()
        # Tell PETSc/SLEPc we are definitely NOT symmetric/Hermitian
        try:
            Ared.setOption(PETSc.Mat.Option.SYMMETRIC, False)
            Ared.setOption(PETSc.Mat.Option.HERMITIAN, False)
        except Exception:
            pass

        # --- EPS: Krylov–Schur on generalized non-Hermitian problem ---
        eps = SLEPc.EPS().create(comm)
        eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        eps.setOperators(Ared, Moo_sub)
        eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

        # Harmonic extraction focuses on interior/left-edge targets
        try:
            eps.setExtraction(SLEPc.EPS.Extraction.HARMONIC)
        except Exception:
            pass

        which_map = {
            "SMALLEST_REAL": SLEPc.EPS.Which.SMALLEST_REAL,
            "LARGEST_REAL":  SLEPc.EPS.Which.LARGEST_REAL,
            "TARGET_REAL":   getattr(SLEPc.EPS.Which, "TARGET_REAL", SLEPc.EPS.Which.TARGET_MAGNITUDE),
        }
        eps.setWhichEigenpairs(which_map.get(which, SLEPc.EPS.Which.TARGET_REAL))
        if which.upper() == "TARGET_REAL":
            try:
                eps.setTarget(float(target_real))
            except Exception:
                pass

        # IMPORTANT: disable two-sided (it would need Ared^T, which we don’t provide)
        try:
            eps.setTwoSided(False)
        except Exception:
            pass
        # Also avoid any balancing tricks that might trigger transpose paths
        try:
            eps.setBalance(SLEPc.EPS.Balance.NONE)
        except Exception:
            pass

        # --- dimensions (auto; user ncv override allowed; consistent) ---
        nev = int(k)
        n_free = Moo_sub.getSize()[0]
        if ncv is None:
            ncv_eff = max(8*nev, 80, int(0.01*n_free))
            ncv_eff = min(ncv_eff, 300)
            ncv_eff = max(ncv_eff, nev + 1)
        else:
            ncv_eff = int(min(max(ncv, nev + 1), n_free))
        mpd_eff = max(1, ncv_eff - nev)
        if verbose:
            print(f"[EPS] dims: nev={nev}, ncv={ncv_eff}, mpd={mpd_eff}, n_free={n_free}")

        try:
            eps.setDimensions(nev=nev, ncv=ncv_eff, mpd=mpd_eff)
        except TypeError:
            eps.setDimensions(nev)

        # Tolerances / iterations
        eps.setTolerances(tol=max(1e-8, tol), max_it=max(4000, max_it))
        eps.setFromOptions()

        # --- Solve ---
        eps.solve()

        nconv = eps.getConverged()
        if verbose:
            try:
                reason = eps.getConvergedReason(); its = eps.getIterationNumber()
                print(f"[EPS] converged={nconv}, iterations={its}, reason={int(reason)}")
            except Exception:
                print(f"[EPS] converged={nconv}")
        if nconv == 0:
            raise RuntimeError("No eigenpairs converged. Try a more negative target_real (e.g. -2), or increase ncv (e.g. 160–200).")

        nev_out = min(nev, nconv)

        # --- Extract eigenpairs, map back to (δΩ, δψ) ---
        eigs, modes_omega, modes_psi = [], [], []
        xr = PETSc.Vec().createMPI(n, comm=comm)
        xi = PETSc.Vec().createMPI(n, comm=comm)
        zpsi = PETSc.Vec().createMPI(ared_ctx.n_psi, comm=comm)
        tpsi = PETSc.Vec().createMPI(ared_ctx.n_psi, comm=comm)
        tmp_full_omega = PETSc.Vec().createMPI(self.mats["n_o"], comm=comm)

        for i in range(nev_out):
            lam = eps.getEigenpair(i, xr, xi)
            eigs.append(lam)

            # δω (Dirichlet-0 on boundary dofs)
            delta_omega = fem.Function(self.Vo)
            try:
                arr = delta_omega.vector.array
            except AttributeError:
                arr = delta_omega.x.array
            arr[:] = 0.0
            xf = xr.getArray(readonly=True)
            arr[self.mats["free_idx"].astype(np.int64)] = xf
            try:
                delta_omega.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                            mode=PETSc.ScatterMode.FORWARD)
            except AttributeError:
                if hasattr(delta_omega.x, "scatter_forward"):
                    delta_omega.x.scatter_forward()

            # δψ = K^{-1} (Mpsi * embed_free_to_full δω)
            tmp_full_omega.set(0.0)
            tmp_full_omega.setValues(self.mats["free_idx"], xf, addv=PETSc.InsertMode.INSERT)
            tmp_full_omega.assemble()
            self.mats["Mpsi"].mult(tmp_full_omega, tpsi)
            zpsi.set(0.0)
            ared_ctx.kspK.solve(tpsi, zpsi)

            delta_psi = fem.Function(self.Vp)
            try:
                arrp = delta_psi.vector.array
            except AttributeError:
                arrp = delta_psi.x.array
            arrp[:] = zpsi.getArray(readonly=True)
            try:
                delta_psi.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                            mode=PETSc.ScatterMode.FORWARD)
            except AttributeError:
                if hasattr(delta_psi.x, "scatter_forward"):
                    delta_psi.x.scatter_forward()

            modes_omega.append(delta_omega)
            modes_psi.append(delta_psi)

        # Sort by smallest real part
        order = np.argsort([np.real(l) for l in eigs])
        eigs = [eigs[j] for j in order]
        modes_omega = [modes_omega[j] for j in order]
        modes_psi = [modes_psi[j] for j in order]
        return eigs, modes_omega, modes_psi
    
    # Export matrices to MATLAB
    # Add this method inside ManifoldEigenSolver
    def export_to_matlab(self, path: str):
        """
        Save the blocks needed to reconstruct A_red in MATLAB:

        A11_sub : (n_free x n_free)
        A12_row : (n_free x n_psi)
        Moo_sub : (n_free x n_free)  [mass on δω]
        Mpsi    : (n_psi  x n_psi)
        K       : (n_psi  x n_psi)   [Neumann stiffness with 1 dof pinned]
        free_idx: (n_free,)          [GLOBAL ω indices that are free]
        n_o, n_p: sizes of ω-full and ψ spaces

        MATLAB will build P = sparse(free_idx, 1:n_free, 1, n_o, n_free)
        and use A_red * x = A11_sub*x + A12_row * (K \ (Mpsi * (P*x))).
        """
        mats = self.mats
        out = {
            "A11_sub": _petsc_to_csr(mats["A11_sub"]),
            "A12_row": _petsc_to_csr(mats["A12_row"]),
            "Moo_sub": _petsc_to_csr(mats["Moo_sub"]),
            "Mpsi":    _petsc_to_csr(mats["Mpsi"]),
            "K":       _petsc_to_csr(mats["K"]),
            # MATLAB is 1-based; store 1-based indices to avoid confusion there
            "free_idx": mats["free_idx"].astype(np.int64) + 1,
            "n_o":      np.array([mats["n_o"]], dtype=np.int64),
            "n_p":      np.array([mats["n_p"]], dtype=np.int64),
        }
        _save_matlab_sparse(path, out)
        try:
            from mpi4py import MPI
            if MPI.COMM_WORLD.rank == 0:
                print(f"[EXPORT] Linearization blocks → {path}")
        except Exception:
            print(f"[EXPORT] Linearization blocks → {path}")



# -------------------------
# Simple plotting helper (reuses your viz)
# -------------------------
def save_eigenmode_plots(mesh, Vpsi, mode_omega, mode_psi=None, idx=0,
                         heatmap_file=None, stream_file=None):
    from viz import save_vorticity_heatmap, save_streamfunction_contours
    if heatmap_file is None:
        heatmap_file = f"eig_{idx:02d}_deltaOmega.png"
    save_vorticity_heatmap(mesh, mode_omega, filename=heatmap_file)
    if mode_psi is not None:
        if stream_file is None:
            stream_file = f"eig_{idx:02d}_deltaPsi_stream.png"
        save_streamfunction_contours(mesh, Vpsi, mode_psi,
                                     filename=stream_file,
                                     n_levels=40, linewidth=0.8)

# Export the Matrices to MATLAB
# --- add to eigs_linearized.py ---
def _petsc_to_csr(A_petsc):
    """Convert PETSc Mat -> SciPy CSR (0-based indices)."""
    import numpy as np
    import scipy.sparse as sp
    A = A_petsc.convert("aij")
    ia, ja, a = A.getValuesCSR()
    ia = ia.copy(); ja = ja.copy(); a = a.copy()
    shape = A.getSize()
    return sp.csr_matrix((a, ja, ia), shape=shape)

def _save_matlab_sparse(path, mats):
    """Save a dict of SciPy sparse matrices + arrays as a .mat file."""
    import scipy.io as sio
    sio.savemat(path, mats, do_compression=True)


