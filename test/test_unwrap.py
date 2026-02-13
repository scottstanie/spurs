"""Tests for phase unwrapping with NumPy, JAX, and CuPy backends."""
import numpy as np
import pytest
from spurs.core import unwrap, make_congruent, HAS_JAX, HAS_CUPY


def _make_ramp(n=512):
    y, x = np.ogrid[-3:3:complex(0, n), -3:3:complex(0, n)]
    phase = np.pi * (x + y)
    return phase, np.angle(np.exp(1j * phase))


def _assert_unwrap_matches(phase, unwrapped, atol=0.1):
    """Check that unwrapped matches original phase up to a constant."""
    p = phase - phase.mean()
    u = unwrapped - unwrapped.mean()
    assert np.allclose(p, u, atol=atol), \
        f"Max difference: {np.max(np.abs(p - u))}"


# --- NumPy backend ---

def test_simple_ramp_numpy():
    phase, wrapped = _make_ramp()
    unwrapped = unwrap(wrapped, max_iters=100, tol=np.pi / 10)
    _assert_unwrap_matches(phase, unwrapped)


def test_quadratic_phase_numpy():
    y, x = np.ogrid[-5:5:128j, -5:5:128j]
    phase = 0.5 * np.pi * (x**2 + y**2)
    wrapped = np.angle(np.exp(1j * phase))
    unwrapped = unwrap(wrapped, max_iters=200, tol=np.pi / 10)
    _assert_unwrap_matches(phase, unwrapped, atol=0.3)


# --- JAX backend ---

@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_simple_ramp_jax():
    phase, wrapped = _make_ramp()
    unwrapped = unwrap(wrapped, backend='jax', max_iters=100, tol=np.pi / 10)
    _assert_unwrap_matches(phase, unwrapped)


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_numpy_match():
    phase, wrapped = _make_ramp(256)
    u_np = unwrap(wrapped, max_iters=100, tol=np.pi / 10)
    u_jax = unwrap(wrapped, backend='jax', max_iters=100, tol=np.pi / 10)
    _assert_unwrap_matches(u_np, u_jax, atol=0.2)


def test_jax_import_error():
    if HAS_JAX:
        pytest.skip("JAX is installed")
    wrapped = np.angle(np.exp(1j * np.pi * np.ones((4, 4))))
    with pytest.raises(ImportError):
        unwrap(wrapped, backend='jax')


# --- CuPy backend ---

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_simple_ramp_cupy():
    phase, wrapped = _make_ramp()
    unwrapped = unwrap(wrapped, backend='cupy', max_iters=100, tol=np.pi / 10)
    _assert_unwrap_matches(phase, unwrapped)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_numpy_match():
    phase, wrapped = _make_ramp(256)
    u_np = unwrap(wrapped, max_iters=100, tol=np.pi / 10)
    u_cp = unwrap(wrapped, backend='cupy', max_iters=100, tol=np.pi / 10)
    _assert_unwrap_matches(u_np, u_cp, atol=0.2)


def test_cupy_import_error():
    if HAS_CUPY:
        pytest.skip("CuPy is installed")
    wrapped = np.angle(np.exp(1j * np.pi * np.ones((4, 4))))
    with pytest.raises(ImportError):
        unwrap(wrapped, backend='cupy')


# --- Congruence ---

def test_make_congruent_basic():
    wrapped = np.array([[0.5, 1.0], [-1.0, 2.0]])
    unwrapped = wrapped + 2 * np.pi * np.array([[1, 3], [2, -1]]) + 0.3
    result = make_congruent(unwrapped, wrapped)
    diff = (result - wrapped) / (2 * np.pi)
    assert np.allclose(diff, np.round(diff))


def test_make_congruent_identity():
    wrapped = np.array([[0.5, -1.0], [2.0, -0.5]])
    unwrapped = wrapped + 2 * np.pi * np.array([[3, -2], [1, 5]], dtype=float)
    assert np.allclose(make_congruent(unwrapped, wrapped), unwrapped)


def test_congruent_flag():
    _, wrapped = _make_ramp(128)
    unwrapped = unwrap(wrapped, max_iters=100, tol=np.pi / 10, congruent=True)
    diff = (unwrapped - wrapped) / (2 * np.pi)
    assert np.allclose(diff, np.round(diff), atol=1e-6)


def test_congruent_preserves_quality():
    phase, wrapped = _make_ramp(256)
    unwrapped = unwrap(wrapped, max_iters=100, tol=np.pi / 10, congruent=True)
    _assert_unwrap_matches(phase, unwrapped, atol=0.2)
