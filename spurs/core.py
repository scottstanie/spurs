"""Core phase unwrapping algorithms for SPURS.

Sparse phase unwrapping using ADMM optimization,
with support for NumPy, JAX, and CuPy backends.
"""

__all__ = [
    'unwrap', 'make_congruent',
    'est_wrapped_gradient', 'p_shrink', 'make_laplace_kernel',
    'make_differentiation_matrices',
]

import numpy as np
from scipy import sparse as sp
from scipy.fft import dctn, idctn

try:
    import jax.numpy as jnp
    from jax.scipy.fft import dct, idct
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

try:
    import cupy as cp
    from cupyx.scipy.fft import dctn as cupy_dctn, idctn as cupy_idctn
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# --- Array-module helpers (work with np, cp, or jnp) ---

def _apply_gradient_x(arr, xp):
    """Forward difference along axis=1 with Neumann BCs."""
    return xp.concatenate([
        arr[:, 1:] - arr[:, :-1],
        xp.zeros((arr.shape[0], 1), dtype=arr.dtype)
    ], axis=1)

def _apply_gradient_y(arr, xp):
    """Forward difference along axis=0 with Neumann BCs."""
    return xp.concatenate([
        arr[1:, :] - arr[:-1, :],
        xp.zeros((1, arr.shape[1]), dtype=arr.dtype)
    ], axis=0)

def _apply_divergence(grad_x, grad_y, xp):
    """Divergence (adjoint of gradient) for Neumann BCs."""
    div_x = xp.concatenate([
        -grad_x[:, :1],
        grad_x[:, :-2] - grad_x[:, 1:-1],
        grad_x[:, -2:-1]
    ], axis=1)
    div_y = xp.concatenate([
        -grad_y[:1, :],
        grad_y[:-2, :] - grad_y[1:-1, :],
        grad_y[-2:-1, :]
    ], axis=0)
    return div_x + div_y


# --- Public helpers (default to numpy, pass xp=cp or xp=jnp for GPU) ---

def est_wrapped_gradient(arr, xp=np, dtype='float32'):
    """Estimate wrapped gradient; wraps result to [-pi, pi]."""
    arr = arr.astype(dtype)
    phi_x = _apply_gradient_x(arr, xp)
    phi_y = _apply_gradient_y(arr, xp)
    phi_x = xp.where(
        xp.abs(phi_x) > xp.pi,
        phi_x - 2 * xp.pi * xp.sign(phi_x), phi_x)
    phi_y = xp.where(
        xp.abs(phi_y) > xp.pi,
        phi_y - 2 * xp.pi * xp.sign(phi_y), phi_y)
    return phi_x, phi_y

def p_shrink(X, xp=np, lmbda=1, p=0, epsilon=0):
    """p-shrinkage operator."""
    mag = xp.sqrt(xp.sum(X ** 2, axis=0))
    nonzero = xp.where(mag == 0.0, 1.0, mag)
    mag = (
        xp.maximum(
            mag - lmbda ** (2.0 - p)
            * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),
            0,
        )
        / nonzero
    )
    return mag * X

def make_laplace_kernel(rows, columns, xp=np, dtype='float32'):
    r"""Eigenvalues of diagonalized Laplacian for Neumann BCs.

    Used for solving the linear system ||D \Phi - phi|| = 0
    via DCT (Numerical Recipes, Section 20.4.1, Eq. 20.4.22).
    """
    xi_y = (
        2 - 2 * xp.cos(xp.pi * xp.arange(rows) / rows)
    ).reshape((-1, 1))
    xi_x = (
        2 - 2 * xp.cos(xp.pi * xp.arange(columns) / columns)
    ).reshape((1, -1))
    eigvals = xi_y + xi_x
    with np.errstate(divide="ignore"):
        K = xp.where(eigvals == 0, 0.0, 1.0 / eigvals)
    return K.astype(dtype)

def make_congruent(unwrapped, wrapped):
    """Adjust unwrapped phase to be congruent with the wrapped phase.

    Rounds to the nearest integer ambiguity so that
        unwrapped_out = wrapped + 2*pi*k
    for integer k at each pixel.
    """
    k = np.round((unwrapped - wrapped) / (2 * np.pi))
    return wrapped + 2 * np.pi * k


# --- Sparse differentiation matrices (utility, not used by unwrap) ---

def make_differentiation_matrices(
    rows, columns, boundary_conditions="neumann", dtype=np.float32
):
    """Generate derivative operators as sparse matrices.

    Source:
    https://github.com/rickchartrand/regularized_differentiation
    """
    bc_opts = ["neumann", "periodic", "dirichlet"]
    bc = boundary_conditions.strip().lower()
    if bc not in bc_opts:
        raise ValueError(f"boundary_conditions must be in {bc_opts}")

    D = sp.diags(
        [-1.0, 1.0], [0, 1], shape=(columns, columns), dtype=dtype
    ).tolil()
    if bc == "neumann":
        D[-1, -1] = 0.0
    elif bc == "periodic":
        D[-1, 0] = 1.0
    Dx = sp.kron(sp.eye(rows, dtype=dtype), D, "csr")

    D = sp.diags(
        [-1.0, 1.0], [0, 1], shape=(rows, rows), dtype=dtype
    ).tolil()
    if bc == "neumann":
        D[-1, -1] = 0.0
    elif bc == "periodic":
        D[-1, 0] = 1.0
    Dy = sp.kron(D, sp.eye(columns, dtype=dtype), "csr")

    return Dx, Dy


# --- Main unwrap function ---

def unwrap(
    f_wrapped,
    phi_x=None,
    phi_y=None,
    max_iters=500,
    tol=np.pi / 5,
    lmbda=1,
    p=0,
    c=1.3,
    dtype="float32",
    debug=False,
    backend="numpy",
    congruent=False,
):
    """Unwrap interferogram phase using ADMM.

    Parameters
    ----------
    f_wrapped : ndarray
        Wrapped phase image (interferogram).
    phi_x, phi_y : ndarray, optional
        Estimates of the x/y-derivatives of the wrapped phase.
    max_iters : int
        Maximum ADMM iterations.
    tol : float
        Convergence tolerance (max change per pixel, radians).
    lmbda : float
        ADMM splitting parameter.
    p : float
        Shrinkage exponent.
    c : float
        Lagrange multiplier acceleration constant.
    dtype : str
        Array dtype.
    debug : bool
        Print iteration diagnostics.
    backend : str
        'numpy', 'jax', or 'cupy'.
    congruent : bool
        If True, snap result to nearest 2*pi*k offset from input.
    """
    # Resolve backend: array module + DCT functions
    if backend == "jax":
        if not HAS_JAX:
            raise ImportError("jax is not installed")
        xp = jnp

        def fwd_dct(x):
            return dct(dct(x, type=2, norm='ortho', axis=0),
                       type=2, norm='ortho', axis=1)

        def inv_dct(x):
            return idct(idct(x, type=2, norm='ortho', axis=1),
                        type=2, norm='ortho', axis=0)
    elif backend == "cupy":
        if not HAS_CUPY:
            raise ImportError("cupy is not installed")
        xp = cp
        fwd_dct = lambda x: cupy_dctn(x, type=2, norm='ortho')
        inv_dct = lambda x: cupy_idctn(x, type=2, norm='ortho')
    else:
        xp = np
        fwd_dct = lambda x: dctn(x, type=2, norm='ortho', workers=-1)
        inv_dct = lambda x: idctn(x, type=2, norm='ortho', workers=-1)

    rows, columns = f_wrapped.shape
    if dtype is not None:
        f_wrapped = f_wrapped.astype(dtype)
    f_wrapped = xp.asarray(f_wrapped)

    if phi_x is None or phi_y is None:
        phi_x, phi_y = est_wrapped_gradient(f_wrapped, xp, dtype=dtype)
    else:
        phi_x, phi_y = xp.asarray(phi_x), xp.asarray(phi_y)

    Lambda_x = xp.zeros_like(phi_x, dtype=dtype)
    Lambda_y = xp.zeros_like(phi_y, dtype=dtype)
    w_x = xp.zeros_like(phi_x, dtype=dtype)
    w_y = xp.zeros_like(phi_y, dtype=dtype)
    F_old = xp.zeros_like(f_wrapped)
    K = make_laplace_kernel(rows, columns, xp, dtype=dtype)

    for iteration in range(max_iters):
        # Solve linear system via DCT
        RHS = _apply_divergence(
            w_x + phi_x - Lambda_x,
            w_y + phi_y - Lambda_y, xp)
        F = inv_dct(fwd_dct(RHS) * K)

        # Gradients of current estimate
        Fx = _apply_gradient_x(F, xp)
        Fy = _apply_gradient_y(F, xp)

        # Shrinkage step
        stacked = xp.stack(
            (Fx - phi_x + Lambda_x, Fy - phi_y + Lambda_y),
            axis=0)
        shrunk = p_shrink(stacked, xp, lmbda=lmbda, p=p)
        w_x, w_y = shrunk[0], shrunk[1]

        # Update Lagrange multipliers
        Lambda_x = Lambda_x + c * (Fx - phi_x - w_x)
        Lambda_y = Lambda_y + c * (Fy - phi_y - w_y)

        change = float(xp.max(xp.abs(F - F_old)))
        if debug:
            print(f"Iteration:{iteration} change={change}")
        if change < tol or np.isnan(change):
            break
        F_old = F

    if debug:
        print(f"Finished after {iteration} with change={change}")

    F = np.asarray(F)
    if congruent:
        F = make_congruent(F, np.asarray(f_wrapped))
    return F
