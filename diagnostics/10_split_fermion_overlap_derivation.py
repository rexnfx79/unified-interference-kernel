#!/usr/bin/env python3
"""
Split-fermion overlap derivation test (wiki: derive-interference-kernel-from-overlaps).

Compares 1D Gaussian wavefunction overlap integrals to the discrete interference kernel
for multiple geometries, up- and down-type Yukawas, and w/sigma ratio stability.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy.optimize import minimize
from kernel import compute_yukawa_matrix


def gaussian_overlap_matrix(
    left_y: np.ndarray,
    right_y: np.ndarray,
    width: float,
) -> np.ndarray:
    """Analytic overlap of normalized Gaussians psi(y) ~ exp(-(y-y0)^2/(2*width^2))."""
    n_l, n_r = len(left_y), len(right_y)
    o = np.zeros((n_l, n_r), dtype=float)
    for i in range(n_l):
        for j in range(n_r):
            d = left_y[i] - right_y[j]
            o[i, j] = np.exp(-(d ** 2) / (4.0 * width ** 2))
    return o


def phase_matrix(
    left_y: np.ndarray,
    right_y: np.ndarray,
    k: float,
    alpha: float,
    eta: float,
) -> np.ndarray:
    """Bilinear phase matching kernel Phi_ij."""
    n_l, n_r = len(left_y), len(right_y)
    phi = np.zeros((n_l, n_r), dtype=float)
    for i in range(n_l):
        for j in range(n_r):
            phi[i, j] = alpha + k * (left_y[i] + right_y[j]) / 2.0 + eta * (left_y[i] - right_y[j])
    return phi


def build_overlap_yukawa(
    left_y: np.ndarray,
    right_y: np.ndarray,
    width: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
) -> np.ndarray:
    mag = gaussian_overlap_matrix(left_y, right_y, width)
    phi = phase_matrix(left_y, right_y, k, alpha, eta)
    return mag * (1.0 + eps * np.exp(1j * phi))


def fit_width_to_kernel(
    left_pos,
    right_pos,
    sigma_target: float,
    k: float,
    alpha: float,
    eta: float,
    eps: float,
    label: str = "Yu",
) -> dict:
    """Find Gaussian width w such that overlap magnitudes best match kernel envelope."""
    left_y = np.array([left_pos[0], left_pos[1], 0.0], dtype=float)
    right_y = np.array(right_pos, dtype=float)
    y_kernel = compute_yukawa_matrix(left_pos, right_pos, sigma_target, k, alpha, eta, eps)
    target_mag = np.abs(y_kernel)

    def loss(log_w):
        w = np.exp(log_w[0])
        mag = gaussian_overlap_matrix(left_y, right_y, w)
        if mag.max() < 1e-15:
            return 1e6
        rel = (mag / mag.max() - target_mag / (np.abs(target_mag).max() + 1e-15)) ** 2
        return float(rel.sum())

    res = minimize(loss, x0=[np.log(sigma_target / np.sqrt(2.0))], method="Nelder-Mead")
    w_opt = float(np.exp(res.x[0]))
    y_overlap = build_overlap_yukawa(left_y, right_y, w_opt, k, alpha, eta, eps)

    mag_corr = np.corrcoef(np.abs(y_overlap).ravel(), np.abs(y_kernel).ravel())[0, 1]
    phase_err = np.mean(np.abs(np.angle(y_overlap / (y_kernel + 1e-15))))

    return {
        "label": label,
        "width": w_opt,
        "sigma_relation": w_opt / sigma_target,
        "magnitude_correlation": float(mag_corr),
        "mean_phase_error_rad": float(phase_err),
        "overlap_loss": float(res.fun),
    }


# Geometries from repo: standard, CSV common, kernel comparison, rigorous validation
GEOMETRIES = [
    {"name": "standard", "Q": (0, 1, 0), "U": (0, 3, 6), "D": (0, 3, 7)},
    {"name": "csv_compact", "Q": (0, 1, 0), "U": (0, 1, 2), "D": (0, 1, 3)},
    {"name": "kernel_comparison", "Q": (0, 1, 3), "U": (2, 4, 5), "D": (0, 3, 6)},
    {"name": "rigorous_validation", "Q": (5, 7, 9), "U": (1, 10, 13), "D": (0, 3, 6)},
]

DEFAULT_PARAMS = dict(sigma_target=4.0, k=1.4, alpha=2.5, eta=2.0, eps=0.15)


def main():
    print("=" * 60)
    print("SPLIT-FERMION OVERLAP DERIVATION TEST")
    print("=" * 60)
    print(f"Kernel params: {DEFAULT_PARAMS}\n")

    all_results = []
    for geom in GEOMETRIES:
        Q, U, D = geom["Q"], geom["U"], geom["D"]
        print(f"--- Geometry: {geom['name']} Q={Q}, U={U}, D={D} ---")
        for sector, right, eps_key in [("Yu", U, "eps"), ("Yd", D, "eps")]:
            result = fit_width_to_kernel(Q, right, **DEFAULT_PARAMS, label=sector)
            result["geometry"] = geom["name"]
            all_results.append(result)
            print(f"  {sector}:")
            for key in ["width", "sigma_relation", "magnitude_correlation", "mean_phase_error_rad"]:
                print(f"    {key}: {result[key]:.6f}")
        print()

    ratios = [r["sigma_relation"] for r in all_results]
    ratio_mean = float(np.mean(ratios))
    ratio_std = float(np.std(ratios))
    ratio_spread = ratio_std / ratio_mean if ratio_mean > 0 else float("inf")

    print("w/sigma ratio stability:")
    print(f"  mean w/σ  = {ratio_mean:.6f}")
    print(f"  std w/σ   = {ratio_std:.6f}")
    print(f"  rel spread = {ratio_spread:.4f}")
    if ratio_spread < 0.05:
        print("  STABLE — w/σ consistent across geometries and sectors")
    else:
        print("  UNSTABLE — w/σ varies; overlap width may be geometry-dependent")

    mag_ok = all(r["magnitude_correlation"] > 0.99 for r in all_results)
    phase_ok = all(r["mean_phase_error_rad"] < 0.05 for r in all_results)
    print("\nOverall:")
    print(f"  Magnitude match (r>0.99 all): {'YES' if mag_ok else 'NO'}")
    print(f"  Phase match (<0.05 rad all): {'YES' if phase_ok else 'NO'}")

    out = os.path.join(os.path.dirname(__file__), "results", "10_split_fermion_overlap.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("Split-fermion overlap derivation test\n\n")
        for r in all_results:
            f.write(f"[{r['geometry']} {r['label']}]\n")
            for k, v in r.items():
                if k not in ("geometry", "label"):
                    f.write(f"{k}: {v}\n")
            f.write("\n")
        f.write(f"w_sigma_mean: {ratio_mean}\n")
        f.write(f"w_sigma_std: {ratio_std}\n")
        f.write(f"w_sigma_rel_spread: {ratio_spread}\n")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
