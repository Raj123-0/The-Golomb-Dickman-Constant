import importlib.util
import mpmath

spec = importlib.util.spec_from_file_location("mod", "MODULE_FILENAME")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_integrand_near_zero_approaches_one():
    """li(x) -> 0 as x -> 0+, so exp(li(x)) -> 1."""
    mpmath.mp.dps = 30
    val = mod.integrand(mpmath.mpf('1e-10'))
    assert abs(val - 1) < mpmath.mpf('1e-5')


def test_integrand_matches_exp_li():
    """integrand(x) == exp(li(x)) by definition."""
    mpmath.mp.dps = 30
    x = mpmath.mpf('0.7')
    assert mod.integrand(x) == mpmath.exp(mpmath.li(x))


def test_golomb_dickman_constant_value():
    """Integral of exp(li(x)) on [0,1] equals the Golomb-Dickman constant."""
    mpmath.mp.dps = 40
    result = mpmath.quad(mod.integrand, [0, 1], method='tanh-sinh')
    known = mpmath.mpf('0.62432998854355087099293638310083724969899')
    assert abs(result - known) < mpmath.mpf('1e-35')


def test_digit_prefix_oeis_convention():
    """Result digits follow OEIS b-file convention (leading 0 included)."""
    mpmath.mp.dps = 40
    result = mpmath.quad(mod.integrand, [0, 1], method='tanh-sinh')
    s = mpmath.nstr(result, 20, strip_zeros=False)
    digits = s.replace('.', '')
    assert digits.startswith('06243299885435508709')


def test_parallel_tanhsinh_class_exists():
    """ParallelTanhSinh is a subclass of mpmath's TanhSinh."""
    assert issubclass(mod.ParallelTanhSinh, mpmath.calculus.quadrature.TanhSinh)


def test_compute_golomb_dickman_signature():
    """compute_golomb_dickman accepts dps and processes kwargs."""
    import inspect
    sig = inspect.signature(mod.compute_golomb_dickman)
    assert 'dps' in sig.parameters
    assert 'processes' in sig.parameters
