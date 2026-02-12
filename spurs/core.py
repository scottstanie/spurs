"""Core phase unwrapping algorithms for SPURS.

This module provides sparse phase unwrapping using ADMM optimization,
with support for NumPy, JAX, and CuPy backends.
"""

__all__ = [
    'est_wrapped_gradient_jax', 'p_shrink_jax',
    'make_laplace_kernel_jax', 'make_differentiation_matrices',
    'est_wrapped_gradient', 'p_shrink', 'make_laplace_kernel',
    'unwrap', 'make_congruent',
]

import numpy as np
from scipy import sparse as sp
from scipy.fft import dctn, idctn
from .loading import load_interferogram

# Optional JAX support
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.fft import dct, idct
    from functools import partial
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jax = None
    jnp = None

# Optional CuPy support
try:
    import cupy as cp
    from cupyx.scipy.fft import dctn as cupy_dctn, idctn as cupy_idctn
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


# --- Generic array-module helpers (work with np, cp, or jnp) ---

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

def _est_wrapped_gradient(arr, xp, dtype='float32'):
    """Estimate wrapped gradient; wraps result to [-pi, pi]."""
    arr = arr.astype(dtype)
    phi_x = _apply_gradient_x(arr, xp)
    phi_y = _apply_gradient_y(arr, xp)
    phi_x = xp.where(xp.abs(phi_x) > xp.pi,
                     phi_x - 2 * xp.pi * xp.sign(phi_x),
                     phi_x)
    phi_y = xp.where(xp.abs(phi_y) > xp.pi,
                     phi_y - 2 * xp.pi * xp.sign(phi_y),
                     phi_y)
    return phi_x, phi_y

def _p_shrink(X, xp, lmbda=1, p=0, epsilon=0):
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

def _make_laplace_kernel(rows, columns, xp, dtype='float32'):
    """Laplacian eigenvalues for Neumann BCs."""
    xi_y = (
        2 - 2 * xp.cos(xp.pi * xp.arange(rows) / rows)
    ).reshape((-1, 1))
    xi_x = (
        2 - 2 * xp.cos(xp.pi * xp.arange(columns) / columns)
    ).reshape((1, -1))
    eigvals = xi_y + xi_x
    K = xp.where(eigvals == 0, 0.0, 1.0 / eigvals)
    return K.astype(dtype)


# --- JAX wrappers (backward-compatible public API) ---

def est_wrapped_gradient_jax(arr, dtype='float32'):
    """Estimate wrapped gradient using JAX."""
    return _est_wrapped_gradient(arr, jnp, dtype=dtype)

def p_shrink_jax(X, lmbda=1, p=0, epsilon=0):
    """JAX version of p-shrinkage."""
    return _p_shrink(X, jnp, lmbda=lmbda, p=p, epsilon=epsilon)

def make_laplace_kernel_jax(rows, columns, dtype='float32'):
    """JAX version of Laplacian kernel."""
    return _make_laplace_kernel(rows, columns, jnp, dtype=dtype)


# --- JAX backend ---

if HAS_JAX:
    @partial(jax.jit, static_argnums=(8, 9, 10))
    def _unwrap_step_jax(F, phi_x, phi_y, Lambda_x, Lambda_y, w_x, w_y, K, lmbda, p, c):
        """Single ADMM iteration (JIT-compiled)"""
        rx = w_x + phi_x - Lambda_x
        ry = w_y + phi_y - Lambda_y
        RHS = _apply_divergence(rx, ry, jnp)

        # JAX's DCT is 1D, so apply to both axes
        rho_hat = dct(dct(RHS, type=2, norm='ortho', axis=0), type=2, norm='ortho', axis=1)
        F = idct(idct(rho_hat * K, type=2, norm='ortho', axis=1), type=2, norm='ortho', axis=0)

        Fx = _apply_gradient_x(F, jnp)
        Fy = _apply_gradient_y(F, jnp)

        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        stacked = jnp.stack((input_x, input_y), axis=0)
        shrunk = _p_shrink(stacked, jnp, lmbda=lmbda, p=p, epsilon=0)
        w_x, w_y = shrunk[0], shrunk[1]

        Lambda_x = Lambda_x + c * (Fx - phi_x - w_x)
        Lambda_y = Lambda_y + c * (Fy - phi_y - w_y)

        return F, Lambda_x, Lambda_y, w_x, w_y

    def unwrap_jax(
        f_wrapped,
        phi_x=None,
        phi_y=None,
        max_iters=500,
        tol=jnp.pi / 5,
        lmbda=1,
        p=0,
        c=1.3,
        dtype="float32",
        debug=False,
    ):
        """JAX-based unwrap with JIT compilation"""
        rows, columns = f_wrapped.shape

        if dtype is None:
            dtype = f_wrapped.dtype
        else:
            f_wrapped = f_wrapped.astype(dtype)

        f_wrapped = jnp.array(f_wrapped)

        if phi_x is None or phi_y is None:
            phi_x, phi_y = _est_wrapped_gradient(f_wrapped, jnp, dtype=dtype)
        else:
            phi_x = jnp.array(phi_x)
            phi_y = jnp.array(phi_y)

        Lambda_x = jnp.zeros_like(phi_x, dtype=dtype)
        Lambda_y = jnp.zeros_like(phi_y, dtype=dtype)
        w_x = jnp.zeros_like(phi_x, dtype=dtype)
        w_y = jnp.zeros_like(phi_y, dtype=dtype)
        F_old = jnp.zeros_like(f_wrapped)

        K = _make_laplace_kernel(rows, columns, jnp, dtype=dtype)

        for iteration in range(max_iters):
            F, Lambda_x, Lambda_y, w_x, w_y = _unwrap_step_jax(
                F_old, phi_x, phi_y, Lambda_x, Lambda_y, w_x, w_y, K, lmbda, p, c
            )

            change = jnp.max(jnp.abs(F - F_old))
            if debug:
                print(f"Iteration:{iteration} change={change}")

            if change < tol or jnp.isnan(change):
                break
            else:
                F_old = F

        if debug:
            print(f"Finished after {iteration} with change={change}")

        return np.array(F)


# --- CuPy backend ---

def unwrap_cupy(
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
):
    """CuPy-based unwrap with GPU acceleration."""
    rows, columns = f_wrapped.shape

    if dtype is None:
        dtype = f_wrapped.dtype
    else:
        f_wrapped = f_wrapped.astype(dtype)

    f_wrapped = cp.asarray(f_wrapped)

    if phi_x is None or phi_y is None:
        phi_x, phi_y = _est_wrapped_gradient(f_wrapped, cp, dtype=dtype)
    else:
        phi_x = cp.asarray(phi_x)
        phi_y = cp.asarray(phi_y)

    Lambda_x = cp.zeros_like(phi_x, dtype=dtype)
    Lambda_y = cp.zeros_like(phi_y, dtype=dtype)
    w_x = cp.zeros_like(phi_x, dtype=dtype)
    w_y = cp.zeros_like(phi_y, dtype=dtype)
    F_old = cp.zeros_like(f_wrapped)

    K = _make_laplace_kernel(rows, columns, cp, dtype=dtype)

    for iteration in range(max_iters):
        rx = w_x + phi_x - Lambda_x
        ry = w_y + phi_y - Lambda_y
        RHS = _apply_divergence(rx, ry, cp)

        # CuPy supports ndim DCT like scipy
        rho_hat = cupy_dctn(RHS, type=2, norm='ortho')
        F = cupy_idctn(rho_hat * K, type=2, norm='ortho')

        Fx = _apply_gradient_x(F, cp)
        Fy = _apply_gradient_y(F, cp)

        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        stacked = cp.stack((input_x, input_y), axis=0)
        shrunk = _p_shrink(stacked, cp, lmbda=lmbda, p=p, epsilon=0)
        w_x, w_y = shrunk[0], shrunk[1]

        Lambda_x = Lambda_x + c * (Fx - phi_x - w_x)
        Lambda_y = Lambda_y + c * (Fy - phi_y - w_y)

        change = float(cp.max(cp.abs(F - F_old)))
        if debug:
            print(f"Iteration:{iteration} change={change}")

        if change < tol or np.isnan(change):
            break
        else:
            F_old = F

    if debug:
        print(f"Finished after {iteration} with change={change}")

    return cp.asnumpy(F)


### Congruence post-processing ###

def make_congruent(unwrapped, wrapped):
    """Adjust unwrapped phase to be congruent with the wrapped phase.

    The ADMM algorithm does not guarantee that the unwrapped phase differs
    from the wrapped phase by an integer multiple of 2pi. This function
    rounds to the nearest integer ambiguity so that
        unwrapped_out = wrapped + 2*pi*k
    for integer k at each pixel.

    Parameters
    ----------
    unwrapped : ndarray
        Unwrapped phase from the solver.
    wrapped : ndarray
        Original wrapped phase (in radians, range [-pi, pi]).

    Returns
    -------
    ndarray
        Adjusted unwrapped phase congruent with the input.
    """
    k = np.round((unwrapped - wrapped) / (2 * np.pi))
    return wrapped + 2 * np.pi * k


# --- NumPy sparse-matrix backend ---

def make_differentiation_matrices(
    rows, columns, boundary_conditions="neumann", dtype=np.float32
):
    """Generate derivative operators as sparse matrices.

    Matrix-vector multiplication is the fastest way to compute derivatives
    of large arrays, particularly for images. This function generates
    the matrices for computing derivatives. If derivatives of the same
    size array will be computed more than once, then it generally is
    faster to compute these arrays once, and then reuse them.

    The three supported boundary conditions are 'neumann' (boundary
    derivative values are zero), 'periodic' (the image ends wrap around
    to beginning), and 'dirichlet' (out-of-bounds elements are zero).
    'neumann' seems to work best for solving the unwrapping problem.

    Source:
    https://github.com/rickchartrand/regularized_differentiation/blob/master/regularized_differentiation/differentiation.py
    """
    bc_opts = ["neumann", "periodic", "dirichlet"]
    bc = boundary_conditions.strip().lower()
    if bc not in bc_opts:
        raise ValueError(f"boundary_conditions must be in {bc_opts}")

    # construct derivative with respect to x (axis=1)
    D = sp.diags([-1.0, 1.0], [0, 1], shape=(columns, columns), dtype=dtype).tolil()

    if boundary_conditions.lower() == bc_opts[0]:  # neumann
        D[-1, -1] = 0.0
    elif boundary_conditions.lower() == bc_opts[1]:  # periodic
        D[-1, 0] = 1.0
    else:
        pass

    S = sp.eye(rows, dtype=dtype)
    Dx = sp.kron(S, D, "csr")

    # construct derivative with respect to y (axis=0)
    D = sp.diags([-1.0, 1.0], [0, 1], shape=(rows, rows), dtype=dtype).tolil()

    if boundary_conditions.lower() == bc_opts[0]:
        D[-1, -1] = 0.0
    elif boundary_conditions.lower() == bc_opts[1]:
        D[-1, 0] = 1.0
    else:
        pass

    S = sp.eye(columns, dtype=dtype)
    Dy = sp.kron(D, S, "csr")

    return Dx, Dy


def est_wrapped_gradient(
    arr, Dx=None, Dy=None, boundary_conditions="neumann", dtype=np.float32
):
    """Estimate the wrapped gradient of `arr` using differential operators `Dx, Dy`
    Adjusts the grad. to be in range [-pi, pi]
    """
    rows, columns = arr.shape
    if Dx is None or Dy is None:
        Dx, Dy = make_differentiation_matrices(
            rows, columns, boundary_conditions=boundary_conditions, dtype=dtype
        )

    phi_x = (Dx @ arr.ravel()).reshape((rows, columns))
    phi_y = (Dy @ arr.ravel()).reshape((rows, columns))
    # Make wrapped adjustmend (eq. (2), (3))
    idxs = np.abs(phi_x) > np.pi
    phi_x[idxs] -= 2 * np.pi * np.sign(phi_x[idxs])
    idxs = np.abs(phi_y) > np.pi
    phi_y[idxs] -= 2 * np.pi * np.sign(phi_y[idxs])
    return phi_x, phi_y


def p_shrink(X, lmbda=1, p=0, epsilon=0):
    """p-shrinkage in 1-D, with mollification."""

    mag = np.sqrt(np.sum(X ** 2, axis=0))
    nonzero = mag.copy()
    nonzero[mag == 0.0] = 1.0
    mag = (
        np.maximum(
            mag
            - lmbda ** (2.0 - p) * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),  # noqa
            0,
        )
        / nonzero
    )

    return mag * X


def make_laplace_kernel(rows, columns, dtype='float32'):
    r"""Generate eigenvalues of diagonalized Laplacian operator

    Used for quickly solving the linear system ||D \Phi - phi|| = 0

    References:
    Numerical recipes, Section 20.4.1, Eq. 20.4.22 is the Neumann case
    or https://elonen.iki.fi/code/misc-notes/neumann-cosine/
    """
    # Note that sign is reversed from numerical recipes eq., since
    # here since our operator discretization sign reversed
    xi_y = (2 - 2 * np.cos(np.pi * np.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * np.cos(np.pi * np.arange(columns) / columns)).reshape((1, -1))
    eigvals = xi_y + xi_x

    with np.errstate(divide="ignore"):
        K = np.nan_to_num(1 / eigvals, posinf=0, neginf=0)
    return K.astype(dtype)


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
    """Unwrap interferogram phase

    Parameters
    ----------
        f_wrapped (ndarray): wrapped phase image (interferogram)
        phi_x (ndarray): estimate of the x-derivative of the wrapped phase
            If not passed, will compute using `est_wrapped_gradient`
        phi_y (ndarray): estimate of the y-derivative of the wrapped phase
            If not passed, will compute using `est_wrapped_gradient`
        max_iters (int): maximum number of ADMM iterations to run
        tol (float): maximum allowed change for any pixel between ADMM iterations
        lmbda (float): splitting parameter of ADMM. Smaller = more stable, Larger = faster convergence.
        p (float): value used in shrinkage operator
        c (float): acceleration constant using in updating lagrange multipliers in ADMM
        dtype: numpy datatype for output
        debug (bool): print diagnostic ADMM information
        backend (str): Backend to use - 'numpy', 'jax', or 'cupy'. Default is 'numpy'.
        congruent (bool): If True, post-process the result so that the unwrapped
            phase differs from the wrapped phase by an integer multiple of 2*pi.
    """
    if backend == "jax":
        if not HAS_JAX:
            raise ImportError(
                "JAX is not installed. Install with: pip install jax"
            )
        result = unwrap_jax(
            f_wrapped, phi_x, phi_y, max_iters, tol,
            lmbda, p, c, dtype, debug,
        )
        if congruent:
            result = make_congruent(result, f_wrapped)
        return result

    if backend == "cupy":
        if not HAS_CUPY:
            raise ImportError(
                "CuPy is not installed. Install with: "
                "pip install cupy-cuda12x"
            )
        result = unwrap_cupy(
            f_wrapped, phi_x, phi_y, max_iters, tol,
            lmbda, p, c, dtype, debug,
        )
        if congruent:
            result = make_congruent(result, f_wrapped)
        return result

    # Original NumPy implementation
    rows, columns = f_wrapped.shape
    num = rows * columns

    if dtype is None:
        dtype = f_wrapped.dtype
    else:
        f_wrapped = f_wrapped.astype(dtype)

    boundary_conditions = "neumann"
    if debug:
        print(f"Making Dx, Dy with BCs={boundary_conditions}")
    Dx, Dy = make_differentiation_matrices(
        *f_wrapped.shape, boundary_conditions=boundary_conditions
    )

    if phi_x is None or phi_y is None:
        phi_x, phi_y = est_wrapped_gradient(f_wrapped, Dx, Dy, dtype=dtype)

    # Lagrange multiplier variables
    Lambda_x = np.zeros_like(phi_x, dtype=dtype)
    Lambda_y = np.zeros_like(phi_y, dtype=dtype)

    # aux. variables for ADMM, holding difference between
    # unwrapped phase gradient and measured gradient from igram
    w_x = np.zeros_like(phi_x, dtype=dtype)
    w_y = np.zeros_like(phi_y, dtype=dtype)

    F_old = np.zeros_like(f_wrapped)

    # Get K ready once for solving linear system
    K = make_laplace_kernel(rows, columns, dtype=dtype)

    for iteration in range(max_iters):

        # update Unwrapped Phase F: solve linear eqn in fourier domain
        # rhs = dx.T @ phi[0].ravel() + dy.T @ phi[1].ravel()
        rx = w_x.ravel() + phi_x.ravel() - Lambda_x.ravel()
        ry = w_y.ravel() + phi_y.ravel() - Lambda_y.ravel()
        RHS = Dx.T * rx + Dy.T * ry
        # Use DCT for neumann:
        rho_hat = dctn(RHS.reshape(rows, columns), type=2, norm='ortho', workers=-1)
        F = idctn(rho_hat * K, type=2, norm='ortho', workers=-1)

        # calculate x, y gradients of new unwrapped phase estimate
        Fx = (Dx @ F.ravel()).reshape(rows, columns)
        Fy = (Dy @ F.ravel()).reshape(rows, columns)

        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        w_x, w_y = p_shrink(
            np.stack((input_x, input_y), axis=0), lmbda=lmbda, p=p, epsilon=0
        )

        # update lagrange multipliers
        Lambda_x += c * (Fx - phi_x - w_x)
        Lambda_y += c * (Fy - phi_y - w_y)

        change = np.max(np.abs(F - F_old))
        if debug:
            print(f"Iteration:{iteration} change={change}")

        if change < tol or np.isnan(change):
            break
        else:
            F_old = F

    if debug:
        print(f"Finished after {iteration} with change={change}")

    if congruent:
        F = make_congruent(F, f_wrapped)
    return F
