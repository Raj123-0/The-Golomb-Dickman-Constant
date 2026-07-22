===============================================================================
PROJECT: The Golomb-Dickman Constant Computation Engine
===============================================================================

OVERVIEW:
Calculates the Golomb-Dickman constant (lambda ≈ 0.6243299885435508...) to high 
precision. This constant arises in random permutation theory and prime factor 
size distributions.

ALGORITHM & IMPLEMENTATION:
- Logarithmic Integral Formulation: Evaluates the definite integral:
    lambda = int_0^1 exp(li(x)) dx
- High-Order Quadrature Integration: Employs parallel Tanh-Sinh (double exponential) 
  and Gauss-Legendre quadrature schemes utilizing gmpy2 and mpmath.
- IPC Zero-Copy Optimization: Quadrature nodes and weights are raw-serialized 
  across worker processes to bypass Python pickling overhead for high precision.
