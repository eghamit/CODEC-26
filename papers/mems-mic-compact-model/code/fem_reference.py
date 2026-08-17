"""Full 2-D finite-element reference model of a capacitive MEMS microphone.

This is the *high-fidelity ground truth* against which the reduced-order compact
model (``compact_model.py``) is verified.  It solves the coupled
electro-mechanical membrane problem on a triangular mesh with no modal
assumption:

    mechanics      -T . grad^2 w = p_ac + p_es(w, V),     w = 0 on the rim
    electrostatics  p_es(w,V) = eps V^2 / (2 (g0 - w)^2)   (attractive)
    capacitance     C(V) = eps . integral dA / (g0 - w)

The nonlinear static equilibrium ``w(V, p)`` is found by Newton iteration with
the exact tangent, the pull-in voltage by arc-length-free bias continuation with
tangent-stiffness monitoring, and the small-signal sensitivity by one solve of
the linearised coupled system.  Linear-triangle (T3) elements are used; the
element Laplacian is the standard constant-gradient stiffness.

The membrane equation is a Poisson problem, so a linear field is reproduced
exactly on any triangle -- the discretisation error is purely the O(h^2)
curvature error, which is why a moderately fine disk mesh converges quickly to
the analytic clamped-membrane results used in the tests.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

EPS0 = 8.8541878128e-12  # vacuum permittivity (F/m)

# 3-point symmetric quadrature on the reference triangle (exact to order 2);
# weights sum to 1/2 (reference-triangle area).  Integral over a physical
# element = sum_g w_g * detJ * f(gp_g), detJ = 2 * area.
_QP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
_QW = np.array([1 / 6, 1 / 6, 1 / 6])
# shape functions N = [1-xi-eta, xi, eta] at the three quadrature points
_NG = np.array([[1 - a - b, a, b] for a, b in _QP])          # (3 gp, 3 node)
# consistent-mass unit block  integral(N_i N_j)/area
_MASS_UNIT = np.array([[2.0, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0


def disk_mesh(radius, n_radial, n_theta):
    """Concentric-ring triangulation of a disk.  Returns (nodes, tris, rim)."""
    nodes = [(0.0, 0.0)]
    thetas = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    for i in range(1, n_radial + 1):
        r = radius * i / n_radial
        for th in thetas:
            nodes.append((r * np.cos(th), r * np.sin(th)))
    nodes = np.asarray(nodes, float)

    def rid(i, k):
        return 1 + (i - 1) * n_theta + (k % n_theta)

    tris = []
    for k in range(n_theta):                       # centre fan
        tris.append((0, rid(1, k), rid(1, k + 1)))
    for i in range(1, n_radial):                   # annular quads
        for k in range(n_theta):
            a, b = rid(i, k), rid(i + 1, k)
            c, d = rid(i + 1, k + 1), rid(i, k + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))
    rim = np.array([rid(n_radial, k) for k in range(n_theta)], int)
    return nodes, np.asarray(tris, int), rim


class DiskMembraneFEM:
    """Coupled electro-mechanical FE model of a clamped circular diaphragm.

    Parameters
    ----------
    radius : float          diaphragm radius (m)
    tension : float         in-plane tension per length T (N/m)
    areal_density : float   rho_s = rho * thickness (kg/m^2)
    gap : float             electrode gap g0 (m)
    permittivity : float    gap permittivity (F/m); default EPS0
    n_radial, n_theta : int mesh resolution
    """

    def __init__(self, radius, tension, areal_density, gap,
                 permittivity=EPS0, n_radial=24, n_theta=48):
        self.radius = float(radius)
        self.tension = float(tension)
        self.areal_density = float(areal_density)
        self.gap = float(gap)
        self.eps = float(permittivity)

        self.nodes, self.tris, self.rim = disk_mesh(radius, n_radial, n_theta)
        self.nn = len(self.nodes)
        self.free = np.setdiff1d(np.arange(self.nn), self.rim)

        self._areas = np.empty(len(self.tris))
        self._B = np.empty((len(self.tris), 2, 3))   # shape-fn gradients
        for e, t in enumerate(self.tris):
            xy = self.nodes[t]
            J = np.array([[xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                          [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]]])
            detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            self._areas[e] = 0.5 * abs(detJ)
            invJT = np.linalg.inv(J).T
            self._B[e] = invJT @ np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])

        self.K = self._assemble_stiffness()          # T * Laplacian
        self.M = self._assemble_mass()               # areal-density mass

    # ---- assembly ------------------------------------------------------
    def _scatter(self, blocks):
        rows, cols, data = [], [], []
        for e, t in enumerate(self.tris):
            Ke = blocks[e]
            for a in range(3):
                for b in range(3):
                    rows.append(t[a]); cols.append(t[b]); data.append(Ke[a, b])
        return sp.csr_matrix((data, (rows, cols)), shape=(self.nn, self.nn))

    def _assemble_stiffness(self):
        blocks = [self.tension * (self._B[e].T @ self._B[e]) * self._areas[e]
                  for e in range(len(self.tris))]
        return self._scatter(blocks)

    def _assemble_mass(self):
        blocks = [self.areal_density * self._areas[e] * _MASS_UNIT
                  for e in range(len(self.tris))]
        return self._scatter(blocks)

    def _pressure_load(self, p):
        f = np.zeros(self.nn)
        for e, t in enumerate(self.tris):
            f[t] += p * self._areas[e] / 3.0
        return f

    # ---- electrostatic load + tangent (nonlinear in w) -----------------
    def _es_load_and_tangent(self, w, V):
        """Consistent nodal ES load f_es and its tangent df_es/dw (both global)."""
        f = np.zeros(self.nn)
        rows, cols, data = [], [], []
        coef = self.eps * V * V
        for e, t in enumerate(self.tris):
            we = w[t]
            detJ = 2.0 * self._areas[e]
            fe = np.zeros(3)
            Ke = np.zeros((3, 3))
            for g in range(3):
                N = _NG[g]
                wq = N @ we
                gap = self.gap - wq
                q = 0.5 * coef / gap**2          # pressure at gp
                dq = coef / gap**3               # d(pressure)/dw at gp
                fe += _QW[g] * detJ * q * N
                Ke += _QW[g] * detJ * dq * np.outer(N, N)
            f[t] += fe
            for a in range(3):
                for b in range(3):
                    rows.append(t[a]); cols.append(t[b]); data.append(Ke[a, b])
        T = sp.csr_matrix((data, (rows, cols)), shape=(self.nn, self.nn))
        return f, T

    # ---- nonlinear static equilibrium ----------------------------------
    def solve_static(self, voltage=0.0, pressure=0.0, w0=None, max_iter=60,
                     tol=1e-12):
        """Newton solve of  K w = f_p + f_es(w,V)  with clamped rim.

        Returns the nodal deflection field ``w`` (m).  Raises ``RuntimeError``
        if Newton fails to converge (used as a pull-in indicator).
        """
        free = self.free
        Kff = self.K[free][:, free].tocsc()
        fp = self._pressure_load(pressure)[free]
        w = np.zeros(self.nn) if w0 is None else w0.copy()
        w[self.rim] = 0.0
        scale = max(self.gap, 1e-12)
        for _ in range(max_iter):
            fes, Tes = self._es_load_and_tangent(w, voltage)
            R = self.K[free][:, free] @ w[free] - fp - fes[free]
            J = (Kff - Tes[free][:, free]).tocsc()
            dw = spla.spsolve(J, -R)
            w[free] += dw
            if np.max(np.abs(dw)) < tol * scale:
                break
        else:
            raise RuntimeError("static Newton did not converge (near/beyond pull-in)")
        # stability: tangent must stay positive definite
        return w

    def tangent_min_eig(self, w, voltage):
        """Smallest eigenvalue of the reduced tangent stiffness at ``w``."""
        free = self.free
        _, Tes = self._es_load_and_tangent(w, voltage)
        J = (self.K[free][:, free] - Tes[free][:, free]).tocsc()
        val = spla.eigsh(J, k=1, which="SA", return_eigenvectors=False)
        return float(val[0])

    def capacitance(self, w):
        """C (F) for a deflection field ``w`` by exact per-element quadrature."""
        C = 0.0
        for e, t in enumerate(self.tris):
            we = w[t]
            detJ = 2.0 * self._areas[e]
            for g in range(3):
                gap = self.gap - _NG[g] @ we
                C += _QW[g] * detJ * self.eps / gap
        return C

    @property
    def area(self):
        return float(self._areas.sum())

    # ---- modal parameters (for compact-model extraction / cross-check) --
    def modal_parameters(self):
        """Extract the fundamental-mode lumped parameters from the FE model.

        Uses the uniform-pressure static shape phi (peak-normalised) -- the mode
        a microphone actually excites -- to define m, k, A_eff and integral(phi^2).
        """
        s = self.solve_static(0.0, 1.0)          # m/Pa
        peak = float(np.max(np.abs(s)))
        phi = s / peak
        m = float(phi @ (self.M @ phi))          # kg
        k = float(phi @ (self.K @ phi))          # N/m
        A_eff = 0.0
        int_phi2 = 0.0
        for e, t in enumerate(self.tris):
            pe = phi[t]
            A_eff += self._areas[e] * pe.mean()
            detJ = 2.0 * self._areas[e]
            for g in range(3):
                val = _NG[g] @ pe
                int_phi2 += _QW[g] * detJ * val * val
        return dict(m=m, k=k, A_eff=A_eff, int_phi2=int_phi2, peak=peak,
                    f0=np.sqrt(k / m) / (2 * np.pi))

    def resonances(self, num=3):
        free = self.free
        Kff = self.K[free][:, free].tocsc()
        Mff = self.M[free][:, free].tocsc()
        vals = spla.eigsh(Kff, k=num, M=Mff, sigma=0.0, which="LM",
                          return_eigenvectors=False)
        return np.sqrt(np.sort(np.clip(vals, 0, None))) / (2 * np.pi)

    # ---- pull-in by bias continuation ----------------------------------
    def pull_in_voltage(self, v_hi=None, steps=40):
        """Largest DC bias with a stable equilibrium (tangent PD), by continuation."""
        if v_hi is None:
            # analytic rigid estimate as an upper bracket, then refine
            mp = self.modal_parameters()
            v_hi = 2.0 * np.sqrt(8 * mp["k"] * self.gap**3
                                 / (27 * self.eps * self.area))
        w = np.zeros(self.nn)
        last_ok, w_ok = 0.0, np.zeros(self.nn)
        hi = v_hi
        for V in np.linspace(0.0, v_hi, steps)[1:]:
            try:
                w = self.solve_static(V, 0.0, w0=w)
                if self.tangent_min_eig(w, V) <= 0:
                    hi = V
                    break
                last_ok, w_ok = V, w.copy()
            except RuntimeError:
                hi = V
                break
        # bisect between last stable V and the first unstable step, always
        # warm-starting Newton from the last converged (stable) field
        lo, w_lo = last_ok, w_ok
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            try:
                w_mid = self.solve_static(mid, 0.0, w0=w_lo)
                stable = self.tangent_min_eig(w_mid, mid) > 0
            except RuntimeError:
                stable = False
            if stable:
                lo, w_lo = mid, w_mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ---- small-signal sensitivity (open circuit, constant charge) ------
    def sensitivity(self, voltage, p_probe=1e-3):
        """Open-circuit dVout/dp (V/Pa) at bias, by a linearised coupled solve.

        A small pressure ``p_probe`` perturbs the biased equilibrium; the biased
        tangent gives dw, hence dC, and the constant-charge relation gives dV.
        """
        w0 = self.solve_static(voltage, 0.0)
        free = self.free
        _, Tes = self._es_load_and_tangent(w0, voltage)
        J = (self.K[free][:, free] - Tes[free][:, free]).tocsc()
        dfp = self._pressure_load(p_probe)[free]
        dw = np.zeros(self.nn)
        dw[free] = spla.spsolve(J, dfp)
        # dC from the linearised capacitance integral about w0
        dC = 0.0
        for e, t in enumerate(self.tris):
            we, dwe = w0[t], dw[t]
            detJ = 2.0 * self._areas[e]
            for g in range(3):
                gap = self.gap - _NG[g] @ we
                dC += _QW[g] * detJ * self.eps * (_NG[g] @ dwe) / gap**2
        C0 = self.capacitance(w0)
        dV = -(voltage / C0) * dC        # constant-charge readout
        return abs(dV / p_probe)
