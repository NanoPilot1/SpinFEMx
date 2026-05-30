import math
import numpy as np
from numba import njit

#This implementation follows the analytic formulation of the double-layer operator by Lindholm (1984). Results were validated by comparing demagnetizing energy against MuMax3/nmag.
# In particular, we use the section 2.2.4 of the thesis tittled Micromagnetic simulations of three dimensional core-shell nanostructures by A. Knittel


@njit(fastmath=True)
def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(fastmath=True)
def _norm(a):
    return math.sqrt(_dot(a, a))


@njit(fastmath=True)
def _det3(a, b, c):
    # det([a b c])
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


@njit(fastmath=True)
def solid_angle_with_atan2(x0, p0, p1, p2):

    tiny = 1e-300
    r0 = p0 - x0
    r1 = p1 - x0
    r2 = p2 - x0

    n0 = _norm(r0) + tiny
    n1 = _norm(r1) + tiny
    n2 = _norm(r2) + tiny

    det = _det3(r0, r1, r2)
    d01 = _dot(r0, r1)
    d12 = _dot(r1, r2)
    d20 = _dot(r2, r0)

    denom = n0 * n1 * n2 + d01 * n2 + d12 * n0 + d20 * n1
    return 2.0 * math.atan2(det, denom)


@njit(fastmath=True)
def _P_log1p(ri, rj, s):
    """
    P = log((ri+rj+s)/(ri+rj-s)), we consider to use log1p for stability:
    (A+s)/(A-s) = 1 + 2s/(A-s) => log1p(2s/(A-s))
    """
    tiny = 1e-300
    A = ri + rj
    denom = A - s
    if denom < tiny:
        denom = tiny
    return math.log1p((2.0 * s) / denom)


@njit(fastmath=True)
def lindholm_weights_precomp(
    x0,
    p0,
    p1,
    p2,
    n_unit,      # unit normal of the triangle
    area,        # area of the triangle
    s_edge,      # (3,) = [|p1-p0|, |p2-p1|, |p0-p2|]
    eta_vecs,    # (3,3): eta_i = n x xi_i 
    gamma_mat,   # (3,3): gamma_{ij} = xi_{i+1} . xi_j 
):
    """
    Returns (w0, w1, w2) associated with vertices (p0, p1, p2) for the double-layer operator.

    """
    tiny = 1e-300

    rho0 = p0 - x0
    rho1 = p1 - x0
    rho2 = p2 - x0

    r0 = _norm(rho0) + tiny
    r1 = _norm(rho1) + tiny
    r2 = _norm(rho2) + tiny

    # Height to the plane of the triangle (constant over the triangle)
    h = _dot(n_unit, rho0)

    # eta_i . rho_i 
    eta0 = _dot(eta_vecs[0], rho0)
    eta1 = _dot(eta_vecs[1], rho1)
    eta2 = _dot(eta_vecs[2], rho2)

    # Edges calculation
    s0 = s_edge[0]
    s1 = s_edge[1]
    s2 = s_edge[2]

    # logs P_i
    P0 = _P_log1p(r0, r1, s0)
    P1 = _P_log1p(r1, r2, s1)
    P2 = _P_log1p(r2, r0, s2)

    Omega = solid_angle_with_atan2(x0, p0, p1, p2)

    # gP_i = sum_j gamma_{i,j} P_j
    gP0 = gamma_mat[0, 0] * P0 + gamma_mat[0, 1] * P1 + gamma_mat[0, 2] * P2
    gP1 = gamma_mat[1, 0] * P0 + gamma_mat[1, 1] * P1 + gamma_mat[1, 2] * P2
    gP2 = gamma_mat[2, 0] * P0 + gamma_mat[2, 1] * P1 + gamma_mat[2, 2] * P2

    A = area + tiny
    c = 1.0 / (8.0 * math.pi * A)

    # Opposite edge convention:

    # w0 uses s1 and eta1; w1 uses s2 and eta2; w2 uses s0 and eta0
    w0 = s1 * c * (eta1 * Omega - h * gP0)
    w1 = s2 * c * (eta2 * Omega - h * gP1)
    w2 = s0 * c * (eta0 * Omega - h * gP2)

    return w0, w1, w2
