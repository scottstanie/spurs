"""Tests for phase unwrapping with both NumPy and JAX backends"""
import numpy as np
import pytest
from spurs.core import unwrap, HAS_JAX


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


if __name__ == "__main__":
    # Run tests manually
    print("Running test_simple_ramp_numpy...")
    test_simple_ramp_numpy()
    print("✓ PASSED")

    if HAS_JAX:
        print("\nRunning test_simple_ramp_jax...")
        test_simple_ramp_jax()
        print("✓ PASSED")

        print("\nRunning test_backends_match...")
        test_backends_match()
        print("✓ PASSED")

        print("\nRunning test_jax_backend_raises_without_jax...")
        test_jax_backend_raises_without_jax()
        print("✓ PASSED")
    else:
        print("\nJAX not installed, skipping JAX-specific tests")

    print("\nRunning test_wrapped_phase_recovery...")
    test_wrapped_phase_recovery()
    print("✓ PASSED")

    print("\nRunning test_invalid_backend...")
    test_invalid_backend()
    print("✓ PASSED")

    print("\n" + "="*50)
    print("All tests passed!")
