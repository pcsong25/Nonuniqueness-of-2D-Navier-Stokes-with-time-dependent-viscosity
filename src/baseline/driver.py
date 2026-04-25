# --- file: driver.py ---
import sys
print("Using Python:", sys.executable)

import numpy as np
from config import SimulationParams, eps_from_R
from mesh import build_disk
from spaces import create_spaces
from boundary_data_fourier import FourierBC
from poisson_neumann import PoissonNeumann
from newton_block import NewtonAssembler
from continuation import ContinuationSolver
from newton_direct import DirectNewtonSolver
from io_utils import write_xdmf
from viz import save_vorticity_heatmap, save_velocity_quiver
from viz import save_mesh_plot
import time
from mpi4py import MPI
from datetime import datetime

from pathlib import Path

# Choose repo-local output directories (edit these paths once)
BASELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASELINE_DIR.parents[1]
SAVE_DIR = (REPO_ROOT / "experiments" / "raw" / "baseline").resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR = BASELINE_DIR
# Optional: timestamped filenames to avoid confusion across runs
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_XDMF = (SAVE_DIR / f"results.xdmf")  # or just "results.xdmf"



def main(make_plots: bool = True):
    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ---- start timer ----
    comm.Barrier()                 # sync ranks (safe even in serial)
    t0 = time.perf_counter()

    ## Start of the main code
    params = SimulationParams(mu=0.8, R=150.0, Rdom=10.0, M=48,
                              tol_newton=1e-9, max_newton=48,
                              sigma_grid=tuple(np.linspace(0.0, 1.0, 21)),
                              h_min=0.06, h_max=0.12)

    mesh, facet_tags = build_disk(params.Rdom, params.h_min, params.h_max)

    save_mesh_plot(mesh, filename=str(PLOT_DIR / "mesh.png"))
    spaces = create_spaces(mesh)
    assert isinstance(spaces, dict), f"create_spaces returned {type(spaces)} (expected dict)"

    # Example Fourier coefficients for \barΩ(θ)
    sig = 1.0
    coeffs = {0: 0.0 + 0.0j, 1: 0.0 + 0.0j, 2: 0.0 + 0.0j, 3: 0.0 + 0.0j, 4: 20.0 + 0j, 5: 0.0 + 0.0j, 
              6: 0.0 + 0.0j, 10: 0.0 + 0.0j, 12: 0.0 + 0.0j, 24: 0.0 + 0.0j, 36: 0.0 + 0.0j}
    # approximate coeffs for step function profile
    # coeffs = {0: np.pi/4 + 0.0j, 1: 0.0 + 0.0j, 2: 0.5 - 0.5j, 3: 0.0 + 0.0j, 4: 0.0 -0.5j, 
    #           5: 0.0 + 0.0j, 6: -1/6 - (1/6)*1.0j, 7: 0.0 + 0.0j, 8: 0.0 + 0.0j, 24: 0.0 + 0.0j, 36: 0.0 + 0.0j}
    # multiply the values in coeffs by sig 
    coeffs = {m: c * sig for m, c in coeffs.items()}

    fourier_bc = FourierBC(params.M, coeffs)

    poisson = PoissonNeumann(mesh, spaces["Vpsi"], facet_tags, outer_tag=1,
                             Rdom=params.Rdom, fourier_bc=fourier_bc, mu=params.mu, R=params.R)

    assembler = NewtonAssembler(spaces["Vomega"], spaces["Vpsi"], facet_tags, outer_tag=1,
                                eps=eps_from_R(params.R, params.Rdom), use_supg=False, tau_scale=1.0)
    
    #Continuation method
    solver = ContinuationSolver(assembler, poisson, params, facet_tags, outer_tag=1, fourier_bc=fourier_bc)
    Omega, psi = solver.run(mesh, spaces)

    # ## Picard + Newton method
    # from warmstart_picard import PicardWarmStart
    # # 1) Warm start (few Picard sweeps)
    # warm = PicardWarmStart(mesh, spaces, poisson, facet_tags, outer_tag=1,
    #                     fourier_bc=fourier_bc, mu=params.mu, R=params.R, Rdom=params.Rdom,
    #                     alpha=0.6, max_sweeps=10, target_drop=1e-2, use_SUPG=False)
    # Omega0, psi0 = warm.run()
    # # 2) Hand off to Newton (start from the warm start)
    # solver = DirectNewtonSolver(assembler, poisson, params,
    #                         linesearch=True, ls_max_steps=6, ls_shrink=0.5, verbose=True)
    # # small helper so Newton starts from (Omega0, psi0)
    # Omega, psi = solver.run_from(mesh, spaces, Omega0, psi0)



    write_xdmf(str(OUT_XDMF), {"Omega": Omega, "psi": psi}, mesh)
    print(f"[SAVE] XDMF written to: {OUT_XDMF}")

    if make_plots:
        from viz import save_vorticity_heatmap
        #from viz import save_velocity_quiver
        from viz import save_streamfunction_contours

        save_streamfunction_contours(mesh, spaces["Vpsi"], psi, filename=str(PLOT_DIR / "streamlines.png"),
                             n_levels=40, linewidth=0.8)

        save_vorticity_heatmap(mesh, Omega, filename=str(PLOT_DIR / "vorticity.png"))
        # pass Vpsi (scalar space) instead of a vector space
        #save_velocity_quiver(mesh, spaces["Vpsi"], psi, filename="velocity.png", stride=4)

    
    ## Print more info about the run
    if MPI.COMM_WORLD.rank == 0:
        print(f"[PARAMS] R = {params.R}, Rdom = {params.Rdom}, mu = {params.mu}")
        print(f"[BC] Fourier coeffs = {coeffs}")  # dict you passed to FourierBC


    # Computed the residues
    from residuals_quick import residuals_quick

    # If you have the pinned gauge dof from your Poisson solver:
    pinned = getattr(poisson, "gauge_dof", None)

    res = residuals_quick(mesh, spaces, Omega, psi,
                        mu=params.mu, R=params.R, Rdom=params.Rdom,
                        facet_tags=facet_tags, outer_tag=1,
                        fourier_bc=fourier_bc, sigma=1.0,
                        pinned_psi_dof=pinned)

    print("\n=== Quick weak residuals (raw l2 of assembled vectors) ===")
    print(f"  eps = {res['meta']['eps']:.3e}, sigma = {res['meta']['sigma']}")
    print(f"  ||R_Omega||_2 = {res['norms']['Omega_l2']:.3e}")
    print(f"  ||R_Psi||_2   = {res['norms']['Psi_l2']:.3e}")


    ## Solve eigenvalue problem for linearized operator
    k = 6
    # from eigs_linearized import ManifoldEigenSolver, save_eigenmode_plots
    # esolver = ManifoldEigenSolver(mesh, spaces, Omega, psi, params, facet_tags, outer_tag=1)

    from eigs_linearized_new import ManifoldEigenSolver, save_modes_plots_real

    es = ManifoldEigenSolver(mesh, spaces, Omega, psi, params, facet_tags, outer_tag=1)
    modes = es.solve(k=k, sigma=-1.0, tol=1e-10, max_it=400, verbose=True)

    for j, m in enumerate(modes[:k], start=1):
        lam = m.lam
        print(f"#{j:2d}: lambda = {lam.real:.15e} {lam.imag:+.15e}j")

    # vorticity + streamlines (requires psi_re computed in your solver):
    save_modes_plots_real(mesh, modes, outdir=str(PLOT_DIR / "eig_plots"), n_plot=k, plot_stream=True)


    es.export_to_matlab(str(SAVE_DIR / "linop_blocks.mat"))

    # # For advection-dominated cases, target a modest negative value and use mass-normalization

    # lams, modes_omega, modes_psi = esolver.solve(
    #     k=21, which="TARGET_REAL", target_real=-1.0,
    #     ncv=None, max_it=2000, tol=1e-8,
    # )

    # print("Eigenvalues (smallest real part first):")
    # for j, lam in enumerate(lams):
    #     print(f"  #{j}: λ = {lam}")

    # for j in range(len(lams)):
    #     save_eigenmode_plots(mesh, spaces["Vpsi"], modes_omega[j], modes_psi[j], idx=j)


    # Export matrices to MATLAB
    #esolver.export_to_matlab("/Users/caltech/Peicong/Nonuniqueness 2D NS with time dep visc/linop_blocks.mat")

    # ---- stop timer ----
    comm.Barrier()
    t1 = time.perf_counter()
    elapsed = t1 - t0

    # (optional) small report of mesh/DoFs
    try:
        ncell = mesh.topology.index_map(mesh.topology.dim).size_global
        nvert = mesh.topology.index_map(0).size_global
        Vo, Vp = spaces["Vomega"], spaces["Vpsi"]
        dofs_o = Vo.dofmap.index_map.size_global * getattr(Vo.dofmap.index_map, "block_size", 1)
        dofs_p = Vp.dofmap.index_map.size_global * getattr(Vp.dofmap.index_map, "block_size", 1)
    except Exception:
        ncell = nvert = dofs_o = dofs_p = -1

    if rank == 0:
        print(f"[TIMING] Total wall time: {elapsed:.2f} s")
        print(f"[MESH] vertices={nvert}, cells={ncell}  |  DoFs: Omega={dofs_o}, psi={dofs_p}")

        # also log to a file
        with open(SAVE_DIR / "run_time.txt", "a") as f:
            f.write(
                f"{datetime.now().isoformat()}  wall={elapsed:.3f}s  "
                f"verts={nvert} cells={ncell}  dofsΩ={dofs_o} dofsψ={dofs_p}\n"
            )


if __name__ == "__main__":
    main()
