from __future__ import annotations

from typing import Tuple
import torch


def _validate_affinity_matrix(A: torch.Tensor) -> None:
    """
    Validate that A is a square 2D affinity matrix.
    """
    if not isinstance(A, torch.Tensor):
        raise TypeError("A must be a torch.Tensor")

    if A.ndim != 2:
        raise ValueError(f"A must be 2D, got shape {tuple(A.shape)}")

    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {tuple(A.shape)}")

    if A.shape[0] < 2:
        raise ValueError("A must be at least 2x2 to compute a Fiedler vector")


def compute_graph_laplacian(
    A: torch.Tensor,
    normalized: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute graph Laplacian from affinity matrix.

    Args:
        A: (B, B) symmetric affinity matrix
        normalized: if True, compute L = I - D^{-1/2} A D^{-1/2}  --- PLAN
                    else compute L = D - A                        --- PAPER
        eps: numerical stability for inverse sqrt

    Returns:
        L: (B, B) graph Laplacian
    """
    _validate_affinity_matrix(A)

    # Force float and symmetrize slightly to reduce numeric drift
    A = A.float()
    A = 0.5 * (A + A.T)

    degree = A.sum(dim=1)

    if normalized:
        inv_sqrt_degree = torch.rsqrt(torch.clamp(degree, min=eps))
        D_inv_sqrt = torch.diag(inv_sqrt_degree)
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        L = I - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        D = torch.diag(degree)
        L = D - A

    # Make sure L is symmetric
    L = 0.5 * (L + L.T)
    return L


def compute_fiedler_vector(
    A: torch.Tensor,
    normalized: bool = True,
    eps: float = 1e-12,
    return_eigenvalues: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the Fiedler vector (the eigenvector of the 2nd smallest eigenvalue).

    Args:
        A: (B, B) affinity matrix
        normalized: whether to use normalized Laplacian
        eps: numerical stability
        return_eigenvalues: if True, also return all eigenvalues

    Returns:
        fiedler_vec: (B,)
        optionally eigenvalues: (B,)
    """
    L = compute_graph_laplacian(A, normalized=normalized, eps=eps)

    # For small B=13, eigh is perfectly fine
    eigenvalues, eigenvectors = torch.linalg.eigh(L)

    # Sort ascending just in case
    idx = torch.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 0th eigenvector is the trivial one; 1st is Fiedler
    fiedler_vec = eigenvectors[:, 1]

    if return_eigenvalues:
        return fiedler_vec, eigenvalues
    return fiedler_vec


def compute_fiedler_magnitude_weights(
    A: torch.Tensor,
    normalized: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Convert Fiedler vector into nonnegative weights using magnitude normalization.

    Args:
        A: (B, B) affinity matrix
        normalized: whether to use normalized Laplacian
        eps: numerical stability

    Returns:
        w_fiedler: (B,), nonnegative weights summing to 1
    """
    fiedler_vec = compute_fiedler_vector(A, normalized=normalized, eps=eps)
    w_fiedler = torch.abs(fiedler_vec)
    w_fiedler = w_fiedler / torch.clamp(w_fiedler.sum(), min=eps)
    return w_fiedler


def check_fiedler_properties(
    A: torch.Tensor,
    normalized: bool = True,
    eps: float = 1e-12,
) -> dict:
    """
    Diagnostic checks for debugging and validation.

    Returns:
        dict with:
            - orthogonality_to_ones
            - weights_sum
            - min_weight
            - max_weight
            - second_eigenvalue
    """
    fiedler_vec, eigenvalues = compute_fiedler_vector(
        A,
        normalized=normalized,
        eps=eps,
        return_eigenvalues=True,
    )
    weights = torch.abs(fiedler_vec)
    weights = weights / torch.clamp(weights.sum(), min=eps)

    ones = torch.ones_like(fiedler_vec)
    orthogonality = torch.dot(fiedler_vec, ones).abs().item()

    return {
        "orthogonality_to_ones": orthogonality,
        "weights_sum": weights.sum().item(),
        "min_weight": weights.min().item(),
        "max_weight": weights.max().item(),
        "second_eigenvalue": eigenvalues[1].item(),
    }


