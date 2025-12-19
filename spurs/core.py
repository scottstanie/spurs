"""Core phase unwrapping algorithms for SPURS.

This module provides sparse phase unwrapping using ADMM optimization,
with support for both NumPy and JAX backends.
"""

__all__ = ['est_wrapped_gradient_jax', 'p_shrink_jax', 'make_laplace_kernel_jax', 'make_differentiation_matrices',
           'est_wrapped_gradient', 'p_shrink', 'make_laplace_kernel', 'unwrap']

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



def _apply_gradient_x_jax(arr):
    """Compute x-gradient using JAX (Neumann boundary conditions)"""
    # Forward difference: arr[:, 1:] - arr[:, :-1]
    grad = jnp.concatenate([
        arr[:, 1:] - arr[:, :-1],
        jnp.zeros((arr.shape[0], 1), dtype=arr.dtype)
    ], axis=1)
    return grad

def _apply_gradient_y_jax(arr):
    """Compute y-gradient using JAX (Neumann boundary conditions)"""
    # Forward difference: arr[1:, :] - arr[:-1, :]
    grad = jnp.concatenate([
        arr[1:, :] - arr[:-1, :],
        jnp.zeros((1, arr.shape[1]), dtype=arr.dtype)
    ], axis=0)
    return grad

def _apply_divergence_jax(grad_x, grad_y):
    """Compute divergence (adjoint of gradient) using JAX

    For Neumann boundary conditions with forward differences:
    - Dx has pattern [-1, 1] along rows, with last column being 0
    - Dx.T (divergence) pattern: first col = -grad[0], middle = grad[i-1] - grad[i], last = grad[-2]
    """
    # Divergence in x-direction (adjoint of forward difference along columns)
    # For each row: [-grad[0], grad[0]-grad[1], grad[1]-grad[2], ..., grad[-2]-grad[-1]]
    # Since grad[:, -1] = 0 (Neumann), last becomes grad[:, -2]
    div_x = jnp.concatenate([
        -grad_x[:, :1],  # First column: -grad[:, 0]
        grad_x[:, :-2] - grad_x[:, 1:-1],  # Middle columns: grad[:, i-1] - grad[:, i]
        grad_x[:, -2:-1]  # Last column: grad[:, -2] (since grad[:, -1] = 0)
    ], axis=1)

    # Divergence in y-direction (adjoint of forward difference along rows)
    div_y = jnp.concatenate([
        -grad_y[:1, :],  # First row: -grad[0, :]
        grad_y[:-2, :] - grad_y[1:-1, :],  # Middle rows: grad[i-1, :] - grad[i, :]
        grad_y[-2:-1, :]  # Last row: grad[-2, :] (since grad[-1, :] = 0)
    ], axis=0)

    return div_x + div_y

def est_wrapped_gradient_jax(arr, dtype=jnp.float32):
    """Estimate wrapped gradient using JAX"""
    arr = arr.astype(dtype)
    phi_x = _apply_gradient_x_jax(arr)
    phi_y = _apply_gradient_y_jax(arr)

    # Wrap to [-pi, pi]
    phi_x = jnp.where(jnp.abs(phi_x) > jnp.pi,
                      phi_x - 2 * jnp.pi * jnp.sign(phi_x),
                      phi_x)
    phi_y = jnp.where(jnp.abs(phi_y) > jnp.pi,
                      phi_y - 2 * jnp.pi * jnp.sign(phi_y),
                      phi_y)
    return phi_x, phi_y



def p_shrink_jax(X, lmbda=1, p=0, epsilon=0):
    """JAX version of p-shrinkage"""
    mag = jnp.sqrt(jnp.sum(X ** 2, axis=0))
    nonzero = jnp.where(mag == 0.0, 1.0, mag)
    mag = (
        jnp.maximum(
            mag - lmbda ** (2.0 - p) * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),
            0,
        )
        / nonzero
    )
    return mag * X

def make_laplace_kernel_jax(rows, columns, dtype='float32'):
    """JAX version of Laplacian kernel"""
    xi_y = (2 - 2 * jnp.cos(jnp.pi * jnp.arange(rows) / rows)).reshape((-1, 1))
    xi_x = (2 - 2 * jnp.cos(jnp.pi * jnp.arange(columns) / columns)).reshape((1, -1))
    eigvals = xi_y + xi_x
    K = jnp.where(eigvals == 0, 0.0, 1.0 / eigvals)
    return K.astype(dtype)



if HAS_JAX:
    @partial(jax.jit, static_argnums=(8, 9, 10))
    def _unwrap_step_jax(F, phi_x, phi_y, Lambda_x, Lambda_y, w_x, w_y, K, lmbda, p, c):
        """Single ADMM iteration (JIT-compiled)"""
        # Solve linear system in Fourier domain
        rx = w_x + phi_x - Lambda_x
        ry = w_y + phi_y - Lambda_y
        RHS = _apply_divergence_jax(rx, ry)

        # DCT for Neumann boundary conditions
        # JAX's DCT is 1D, so apply to both axes
        rho_hat = dct(dct(RHS, type=2, norm='ortho', axis=0), type=2, norm='ortho', axis=1)
        F = idct(idct(rho_hat * K, type=2, norm='ortho', axis=1), type=2, norm='ortho', axis=0)

        # Compute gradients
        Fx = _apply_gradient_x_jax(F)
        Fy = _apply_gradient_y_jax(F)

        # Shrinkage step
        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        stacked = jnp.stack((input_x, input_y), axis=0)
        shrunk = p_shrink_jax(stacked, lmbda=lmbda, p=p, epsilon=0)
        w_x, w_y = shrunk[0], shrunk[1]

        # Update Lagrange multipliers
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

        # Convert to JAX arrays
        f_wrapped = jnp.array(f_wrapped)

        if phi_x is None or phi_y is None:
            phi_x, phi_y = est_wrapped_gradient_jax(f_wrapped, dtype=dtype)
        else:
            phi_x = jnp.array(phi_x)
            phi_y = jnp.array(phi_y)

        # Initialize variables
        Lambda_x = jnp.zeros_like(phi_x, dtype=dtype)
        Lambda_y = jnp.zeros_like(phi_y, dtype=dtype)
        w_x = jnp.zeros_like(phi_x, dtype=dtype)
        w_y = jnp.zeros_like(phi_y, dtype=dtype)
        F_old = jnp.zeros_like(f_wrapped)

        # Precompute Laplacian kernel
        K = make_laplace_kernel_jax(rows, columns, dtype=dtype)

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

        # Convert back to numpy for consistency
        return np.array(F)



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
        backend (str): Backend to use - 'numpy' or 'jax'. Default is 'numpy'.
    """
    if backend == "jax":
        if not HAS_JAX:
            raise ImportError("JAX is not installed. Install with: pip install jax jaxlib")
        return unwrap_jax(f_wrapped, phi_x, phi_y, max_iters, tol, lmbda, p, c, dtype, debug)

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
    return F
