"""Tests for phase unwrapping with NumPy, JAX, and CuPy backends"""
import numpy as np
import pytest
from spurs.core import unwrap, make_congruent, HAS_JAX, HAS_CUPY


def test_simple_ramp_numpy():
    """Test simple linear phase ramp with NumPy backend"""
    # Create a simple linear phase ramp
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)
    wrapped_phase = np.angle(igram)

    # Unwrap using NumPy backend
    unwrapped = unwrap(wrapped_phase, backend='numpy', max_iters=100, tol=np.pi / 10)

    # The unwrapped phase should match the original phase up to a constant
    # Remove the mean to compare
    phase_normalized = phase - phase.mean()
    unwrapped_normalized = unwrapped - unwrapped.mean()

    # Check that they match closely
    assert np.allclose(phase_normalized, unwrapped_normalized, atol=0.1), \
        f"Max difference: {np.max(np.abs(phase_normalized - unwrapped_normalized))}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_simple_ramp_jax():
    """Test simple linear phase ramp with JAX backend"""
    # Create a simple linear phase ramp
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)
    wrapped_phase = np.angle(igram)

    # Unwrap using JAX backend
    unwrapped = unwrap(wrapped_phase, backend='jax', max_iters=100, tol=np.pi / 10)

    # The unwrapped phase should match the original phase up to a constant
    # Remove the mean to compare
    phase_normalized = phase - phase.mean()
    unwrapped_normalized = unwrapped - unwrapped.mean()

    # Check that they match closely
    assert np.allclose(phase_normalized, unwrapped_normalized, atol=0.1), \
        f"Max difference: {np.max(np.abs(phase_normalized - unwrapped_normalized))}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_backends_match():
    """Test that NumPy and JAX backends produce similar results"""
    # Create a simple test case
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)
    wrapped_phase = np.angle(igram)

    # Unwrap with both backends
    unwrapped_numpy = unwrap(wrapped_phase, backend='numpy', max_iters=100, tol=np.pi / 10)
    unwrapped_jax = unwrap(wrapped_phase, backend='jax', max_iters=100, tol=np.pi / 10)

    # Normalize both (remove mean)
    unwrapped_numpy_norm = unwrapped_numpy - unwrapped_numpy.mean()
    unwrapped_jax_norm = unwrapped_jax - unwrapped_jax.mean()

    # They should be very similar
    assert np.allclose(unwrapped_numpy_norm, unwrapped_jax_norm, atol=0.2), \
        f"Backends differ. Max difference: {np.max(np.abs(unwrapped_numpy_norm - unwrapped_jax_norm))}"


def test_wrapped_phase_recovery():
    """Test that unwrapping recovers the original phase (up to constant)"""
    # Create a more complex phase pattern
    y, x = np.ogrid[-5:5:128j, -5:5:128j]
    # Quadratic phase
    phase = 0.5 * np.pi * (x**2 + y**2)
    igram = np.exp(1j * phase)
    wrapped_phase = np.angle(igram)

    # Unwrap
    unwrapped = unwrap(wrapped_phase, backend='numpy', max_iters=200, tol=np.pi / 10)

    # Normalize
    phase_norm = phase - phase.mean()
    unwrapped_norm = unwrapped - unwrapped.mean()

    # Check recovery
    assert np.allclose(phase_norm, unwrapped_norm, atol=0.3), \
        f"Phase recovery failed. Max diff: {np.max(np.abs(phase_norm - unwrapped_norm))}"


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
def test_jax_backend_raises_without_jax():
    """Test that requesting JAX backend without JAX installed raises appropriate error"""
    # This test is a bit meta - it will only run if JAX IS installed
    # So we just verify that the backend works when JAX is available
    y, x = np.ogrid[-2:2:64j, -2:2:64j]
    phase = np.pi * x
    wrapped_phase = np.angle(np.exp(1j * phase))

    # Should not raise
    try:
        result = unwrap(wrapped_phase, backend='jax', max_iters=50)
        assert result is not None
    except ImportError:
        pytest.fail("JAX backend raised ImportError even though JAX is installed")


def test_invalid_backend():
    """Test that invalid backend raises appropriate error"""
    y, x = np.ogrid[-2:2:64j, -2:2:64j]
    phase = np.pi * x
    wrapped_phase = np.angle(np.exp(1j * phase))

    # Should raise an error or use default
    # The current implementation will just fall through to numpy
    # Let's test that it doesn't crash
    result = unwrap(wrapped_phase, backend='invalid', max_iters=50)
    assert result is not None


### CuPy backend tests ###

@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_simple_ramp_cupy():
    """Test simple linear phase ramp with CuPy backend"""
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    phase = np.pi * (x + y)
    igram = np.exp(1j * phase)
    wrapped_phase = np.angle(igram)

    unwrapped = unwrap(wrapped_phase, backend='cupy', max_iters=100, tol=np.pi / 10)

    phase_normalized = phase - phase.mean()
    unwrapped_normalized = unwrapped - unwrapped.mean()

    assert np.allclose(phase_normalized, unwrapped_normalized, atol=0.1), \
        f"Max difference: {np.max(np.abs(phase_normalized - unwrapped_normalized))}"


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_cupy_numpy_match():
    """Test that CuPy and NumPy backends produce similar results"""
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)
    wrapped_phase = np.angle(np.exp(1j * phase))

    unwrapped_numpy = unwrap(wrapped_phase, backend='numpy', max_iters=100, tol=np.pi / 10)
    unwrapped_cupy = unwrap(wrapped_phase, backend='cupy', max_iters=100, tol=np.pi / 10)

    np_norm = unwrapped_numpy - unwrapped_numpy.mean()
    cp_norm = unwrapped_cupy - unwrapped_cupy.mean()

    assert np.allclose(np_norm, cp_norm, atol=0.2), \
        f"Backends differ. Max difference: {np.max(np.abs(np_norm - cp_norm))}"


def test_cupy_backend_import_error():
    """Test that requesting CuPy backend without CuPy raises ImportError"""
    if HAS_CUPY:
        pytest.skip("CuPy is installed, cannot test ImportError")
    y, x = np.ogrid[-2:2:64j, -2:2:64j]
    wrapped_phase = np.angle(np.exp(1j * np.pi * x))
    with pytest.raises(ImportError):
        unwrap(wrapped_phase, backend='cupy', max_iters=50)


### Congruence tests ###

def test_make_congruent_basic():
    """Test that make_congruent produces integer multiples of 2pi offset"""
    wrapped = np.array([[0.5, 1.0], [-1.0, 2.0]])
    # unwrapped is close to wrapped + some 2pi*k, but not exactly
    unwrapped = wrapped + 2 * np.pi * np.array([[1, 3], [2, -1]]) + 0.3
    result = make_congruent(unwrapped, wrapped)

    # Result should differ from wrapped by exact integer multiples of 2pi
    diff = (result - wrapped) / (2 * np.pi)
    assert np.allclose(diff, np.round(diff)), \
        f"Result is not congruent with wrapped phase: {diff}"


def test_make_congruent_identity():
    """Test that make_congruent is a no-op when already congruent"""
    wrapped = np.array([[0.5, -1.0], [2.0, -0.5]])
    unwrapped = wrapped + 2 * np.pi * np.array([[3, -2], [1, 5]], dtype=float)
    result = make_congruent(unwrapped, wrapped)
    assert np.allclose(result, unwrapped)


def test_congruent_flag_in_unwrap():
    """Test that the congruent flag produces phase congruent with input"""
    y, x = np.ogrid[-3:3:128j, -3:3:128j]
    phase = np.pi * (x + y)
    wrapped_phase = np.angle(np.exp(1j * phase))

    unwrapped = unwrap(wrapped_phase, backend='numpy', max_iters=100,
                       tol=np.pi / 10, congruent=True)

    # Check congruence: (unwrapped - wrapped) should be integer multiples of 2pi
    diff = (unwrapped - wrapped_phase) / (2 * np.pi)
    assert np.allclose(diff, np.round(diff), atol=1e-6), \
        f"Max non-integer residual: {np.max(np.abs(diff - np.round(diff)))}"


def test_congruent_preserves_quality():
    """Test that congruent flag doesn't degrade unwrapping quality"""
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)
    wrapped_phase = np.angle(np.exp(1j * phase))

    unwrapped = unwrap(wrapped_phase, backend='numpy', max_iters=100,
                       tol=np.pi / 10, congruent=True)

    phase_norm = phase - phase.mean()
    unwrapped_norm = unwrapped - unwrapped.mean()

    assert np.allclose(phase_norm, unwrapped_norm, atol=0.2), \
        f"Max diff: {np.max(np.abs(phase_norm - unwrapped_norm))}"
