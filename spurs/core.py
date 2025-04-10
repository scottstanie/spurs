# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_core.ipynb (unless otherwise specified).

__all__ = [
    "make_differentiation_matrices",
    "est_wrapped_gradient",
    "p_shrink",
    "make_laplace_kernel",
    "unwrap",
]

# Cell
import numpy as np
from scipy import sparse as sp
from scipy.fft import dctn, idctn


# Cell
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


# Cell
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


import jax
import jax.numpy as jnp


def dx(arr, mode="constant"):
    return jnp.pad(jnp.diff(arr, axis=1), ((0, 0), (0, 1)), mode=mode)


def dy(arr, mode="constant"):
    return jnp.pad(jnp.diff(arr, axis=0), ((0, 1), (0, 0)), mode=mode)


def dx_transpose(arr):
    """Compute transposed derivative along x-axis (columns) with boundary conditions."""
    # Adjoint of the differentiation operator is -Dx
    d_arr = -dx(arr)
    # array([[ 0.,  1.,  2.,  3.],
    #        [ 4.,  5.,  6.,  7.],
    #        [ 8.,  9., 10., 11.],
    #        [12., 13., 14., 15.]], dtype=float32)
    # (Dx3.T @ w.ravel()).reshape(4, 4)
    # array([[  0.,  -1.,  -1.,   2.],
    #        [ -4.,  -1.,  -1.,   6.],
    #        [ -8.,  -1.,  -1.,  10.],
    #        [-12.,  -1.,  -1.,  14.]], dtype=float32)
    d_arr = d_arr.at[:, 0].set(-arr[:, 0])
    return d_arr.at[:, -1].set(arr[:, -2])


def dy_transpose(arr, boundary_condition="neumann"):
    """Compute the negative of the forward difference along y-axis (rows) with boundary conditions."""
    # array([[ 0.,  1.,  2.,  3.],
    #        [ 4.,  5.,  6.,  7.],
    #        [ 8.,  9., 10., 11.],
    #        [12., 13., 14., 15.]], dtype=float32)
    # (Dy3.T @ w.ravel()).reshape(4, 4)
    # array([[ 0., -1., -2., -3.],
    #        [-4., -4., -4., -4.],
    #        [-4., -4., -4., -4.],
    #        [ 8.,  9., 10., 11.]], dtype=float32)
    d_arr = -dy(arr)
    d_arr = d_arr.at[0, :].set(-arr[0, :])
    return d_arr.at[-1, :].set(arr[-2, :])


@jax.jit
def est_wrapped_gradient_jax(arr):
    """Estimate the wrapped gradient of `arr` using differential operators `Dx, Dy`
    Adjusts the grad. to be in range [-pi, pi]
    """
    phi_x = dx(arr)
    phi_y = dy(arr)
    # Make wrapped adjustment (eq. (2), (3))
    idxs = jnp.abs(phi_x) > jnp.pi
    fillval = phi_x - 2 * jnp.pi * jnp.sign(phi_x)
    phi_out_x = jnp.where(idxs, fillval, phi_x)

    idxs = jnp.abs(phi_y) > jnp.pi
    fillval = phi_y - 2 * jnp.pi * jnp.sign(phi_y)
    phi_out_y = jnp.where(idxs, fillval, phi_y)

    return phi_out_x, phi_out_y


# Cell
def p_shrink(X, lmbda=1, p=0, epsilon=0):
    """p-shrinkage in 1-D, with mollification."""

    mag = np.sqrt(np.sum(X**2, axis=0))
    nonzero = mag.copy()
    nonzero[mag == 0.0] = 1.0
    mag = (
        np.maximum(
            mag - lmbda ** (2.0 - p) * (nonzero**2 + epsilon) ** (p / 2.0 - 0.5),  # noqa
            0,
        )
        / nonzero
    )

    return mag * X


# Cell
def make_laplace_kernel(rows, columns, dtype="float32"):
    """Generate eigenvalues of diagonalized Laplacian operator

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


from pathlib import Path
from numpy.typing import NDArray


def tophu_callback(
    igram: NDArray[np.complexfloating],
    coherence: NDArray[np.floating],
    nlooks: float,
    scratchdir: Path,
    **kwargs,
):
    """Work around the not needing nlooks/corr ."""
    unw = unwrap(np.angle(igram).astype("float32"))  # , **kwargs)
    return unw, np.ones(unw.shape, dtype="uint32")


# Cell
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
    #     boundary_conditions="neumann",
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
    """
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
        rho_hat = dctn(RHS.reshape(rows, columns), type=2, norm="ortho", workers=-1)
        F = idctn(rho_hat * K, type=2, norm="ortho", workers=-1)

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


def p_shrink_jax(input_x, input_y, lmbda=1, p=0, epsilon=0):
    """p-shrinkage in 1-D, with mollification."""
    # mag = jnp.sqrt(jnp.sum(X**2, axis=0))
    mag = jnp.sqrt(input_x**2 + input_y**2)
    nonzero = jnp.where(mag == 0, 1.0, mag)
    mag = (
        jnp.maximum(
            mag - lmbda ** (2.0 - p) * (nonzero**2 + epsilon) ** (p / 2.0 - 0.5),
            0,
        )
        / nonzero
    )

    return mag * input_x, mag * input_y


# Cell
# @jax.jit
def unwrap_jax(
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
    #     boundary_conditions="neumann",
):
    from jax.scipy.fft import dctn, idctn
    import jax.numpy as np

    rows, columns = f_wrapped.shape

    phi_x, phi_y = est_wrapped_gradient_jax(f_wrapped)

    # Lagrange multiplier variables
    Lambda_x = np.zeros_like(phi_x, dtype=dtype)
    Lambda_y = np.zeros_like(phi_y, dtype=dtype)

    # aux. variables for ADMM, holding difference between
    # unwrapped phase gradient and measured gradient from igram
    w_x = np.zeros_like(phi_x, dtype=dtype)
    w_y = np.zeros_like(phi_y, dtype=dtype)

    F_old = np.zeros_like(f_wrapped)
    F_init = np.ones_like(f_wrapped)

    # Get K ready once for solving linear system
    K = make_laplace_kernel(rows, columns, dtype=dtype)

    # for iteration in range(max_iters):
    def loop_body(val):
        iteration, F, F_old, Lambda_x, Lambda_y, w_x, w_y = val
        F_old = F

        # update Unwrapped Phase F: solve linear eqn in fourier domain
        # rhs = dx.T @ phi[0].ravel() + dy.T @ phi[1].ravel()
        # rx = w_x.ravel() + phi_x.ravel() - Lambda_x.ravel()
        # ry = w_y.ravel() + phi_y.ravel() - Lambda_y.ravel()
        rx = w_x + phi_x - Lambda_x
        ry = w_y + phi_y - Lambda_y
        RHS = dx_transpose(rx) + dy_transpose(ry)
        # Use DCT for neumann:
        rho_hat = dctn(RHS, type=2, norm="ortho")
        F = idctn(rho_hat * K, type=2, norm="ortho")

        # calculate x, y gradients of new unwrapped phase estimate
        Fx = dx(F)
        Fy = dy(F)

        input_x = Fx - phi_x + Lambda_x
        input_y = Fy - phi_y + Lambda_y
        w_x, w_y = p_shrink_jax(input_x, input_y, lmbda=lmbda, p=p, epsilon=0)

        # update lagrange multipliers
        Lambda_x += c * (Fx - phi_x - w_x)
        Lambda_y += c * (Fy - phi_y - w_y)

        # if debug:
        change = np.max(np.abs(F - F_old))
        print(f"Iteration:{iteration} change={change}")

        # if change < tol or np.isnan(change):
        #     break
        # else:
        #     F_old = F
        F_old = jax.lax.cond(change < tol, lambda _: F, lambda _: F_old, None)
        return iteration + 1, F, F_old, Lambda_x, Lambda_y, w_x, w_y

    def loop_cond(val):
        iteration, F, F_old, Lambda_x, Lambda_y, w_x, w_y = val
        change = jnp.max(jnp.abs(F - F_old))
        print(change)
        return (change >= tol) & (iteration < max_iters)

    init_val = (0, F_init, F_old, Lambda_x, Lambda_y, w_x, w_y)
    iteration_final, F_final, F_last, *_ = jax.lax.while_loop(
        loop_cond, loop_body, init_val
    )
    # if debug:
    change = jnp.max(jnp.abs(F_final - F_last))
    print(f"Finished after {iteration_final.block_until_ready()} with change={change}")
    return F_final
