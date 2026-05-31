"""Pure-Python linear algebra for deterministic ridge regression.

No numpy dependency.  All operations use Python float (float64).
Rounding is NOT applied inside operations — only at output boundaries.

Key function: solve_ridge(X, Y, lambda_reg) solves via Cholesky
decomposition (numerically stable for symmetric positive definite matrices).

Reference: Golub & Van Loan, "Matrix Computations", 4th ed.
"""

from __future__ import annotations

import math
from typing import List

# Type alias: matrix = list of rows, each row = list of floats
Matrix = List[List[float]]


# ---------------------------------------------------------------------------
# Basic matrix operations
# ---------------------------------------------------------------------------

def mat_zeros(rows: int, cols: int) -> Matrix:
    """Create a rows×cols zero matrix."""
    return [[0.0] * cols for _ in range(rows)]


def mat_identity(n: int) -> Matrix:
    """Create an n×n identity matrix."""
    m = mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def mat_transpose(A: Matrix) -> Matrix:
    """Transpose matrix A."""
    rows = len(A)
    cols = len(A[0]) if rows > 0 else 0
    return [[A[i][j] for i in range(rows)] for j in range(cols)]


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    """Multiply A (m×n) by B (n×p) → C (m×p)."""
    m = len(A)
    n = len(A[0]) if m > 0 else 0
    p = len(B[0]) if len(B) > 0 else 0
    C = mat_zeros(m, p)
    for i in range(m):
        for k in range(n):
            a_ik = A[i][k]
            if a_ik == 0.0:
                continue
            for j in range(p):
                C[i][j] += a_ik * B[k][j]
    return C


def mat_add(A: Matrix, B: Matrix) -> Matrix:
    """Element-wise A + B."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def mat_scale(A: Matrix, s: float) -> Matrix:
    """Scalar multiply: s * A."""
    return [[A[i][j] * s for j in range(len(A[0]))]
            for i in range(len(A))]


def vec_dot(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Cholesky decomposition
# ---------------------------------------------------------------------------

def cholesky(A: Matrix) -> Matrix:
    """Cholesky decomposition: A = L·L^T.

    A must be symmetric positive definite.
    Returns lower-triangular L.

    Raises ValueError if A is not positive definite.
    """
    n = len(A)
    L = mat_zeros(n, n)

    for j in range(n):
        # Diagonal element
        s = A[j][j]
        for k in range(j):
            s -= L[j][k] * L[j][k]
        if s <= 0.0:
            raise ValueError(
                f"Matrix is not positive definite (pivot {j}: {s:.6e}). "
                f"Try increasing lambda_reg."
            )
        L[j][j] = math.sqrt(s)

        # Below-diagonal elements
        inv_ljj = 1.0 / L[j][j]
        for i in range(j + 1, n):
            s = A[i][j]
            for k in range(j):
                s -= L[i][k] * L[j][k]
            L[i][j] = s * inv_ljj

    return L


def solve_triangular_lower(L: Matrix, B: Matrix) -> Matrix:
    """Solve L·X = B for X, where L is lower-triangular.

    Forward substitution.  B is (n × m), returns X (n × m).
    """
    n = len(L)
    m = len(B[0]) if len(B) > 0 else 0
    X = mat_zeros(n, m)

    for j in range(m):
        for i in range(n):
            s = B[i][j]
            for k in range(i):
                s -= L[i][k] * X[k][j]
            X[i][j] = s / L[i][i]

    return X


def solve_triangular_upper(U: Matrix, B: Matrix) -> Matrix:
    """Solve U·X = B for X, where U is upper-triangular.

    Back substitution.  B is (n × m), returns X (n × m).
    """
    n = len(U)
    m = len(B[0]) if len(B) > 0 else 0
    X = mat_zeros(n, m)

    for j in range(m):
        for i in range(n - 1, -1, -1):
            s = B[i][j]
            for k in range(i + 1, n):
                s -= U[i][k] * X[k][j]
            X[i][j] = s / U[i][i]

    return X


# ---------------------------------------------------------------------------
# Ridge regression solver
# ---------------------------------------------------------------------------

def solve_ridge(
    X: Matrix,
    Y: Matrix,
    lambda_reg: float = 1e-4,
) -> Matrix:
    """Solve ridge regression: W = Y·X^T · (X·X^T + λI)^{-1}.

    Uses Cholesky decomposition for numerical stability.

    Args:
        X: Feature matrix (d × N) — d features, N samples.
        Y: Target matrix (k × N) — k outputs, N samples.
        lambda_reg: Regularization parameter (must be > 0).

    Returns:
        W: Weight matrix (k × d).
    """
    if lambda_reg <= 0:
        raise ValueError("lambda_reg must be positive for Cholesky stability")

    d = len(X)      # feature dimension
    N = len(X[0]) if d > 0 else 0
    k = len(Y)      # output dimension

    # 1. A = X·X^T + λI  (d × d, symmetric positive definite)
    Xt = mat_transpose(X)
    A = mat_mul(X, Xt)
    for i in range(d):
        A[i][i] += lambda_reg

    # 2. B = Y·X^T  (k × d)
    B = mat_mul(Y, Xt)

    # 3. Solve A · W^T = B^T for W^T, then transpose to get W
    #    Equivalently: solve W · A^T = B, but since A is symmetric: W · A = B
    #    So we need W such that A · W^T = B^T
    Bt = mat_transpose(B)  # (d × k)

    # Cholesky: A = L · L^T
    L = cholesky(A)
    Lt = mat_transpose(L)

    # Solve L · Z = B^T  (forward substitution)
    Z = solve_triangular_lower(L, Bt)

    # Solve L^T · W^T = Z  (back substitution)
    Wt = solve_triangular_upper(Lt, Z)

    # W = (W^T)^T  → (k × d)
    W = mat_transpose(Wt)

    return W


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean(xs: List[float]) -> float:
    """Arithmetic mean."""
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def variance(xs: List[float]) -> float:
    """Population variance."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std(xs: List[float]) -> float:
    """Population standard deviation."""
    return math.sqrt(variance(xs))


def mae(predicted: List[float], actual: List[float]) -> float:
    """Mean Absolute Error."""
    if not predicted:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


def r_squared(predicted: List[float], actual: List[float]) -> float:
    """Coefficient of determination (R²)."""
    if len(actual) < 2:
        return 0.0
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    m = mean(actual)
    ss_tot = sum((a - m) ** 2 for a in actual)
    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0
    return 1.0 - ss_res / ss_tot


def accuracy(predicted: List[int], actual: List[int]) -> float:
    """Exact-match accuracy."""
    if not predicted:
        return 0.0
    return sum(1 for p, a in zip(predicted, actual) if p == a) / len(predicted)
