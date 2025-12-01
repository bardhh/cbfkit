from typing import List, Tuple

import numpy as np
from scipy.stats import norm


def _pdf_or_ones(values, std):
    if std == 0:
        return np.ones((len(values), 1))
    else:
        return norm.pdf(values, 0, std).reshape(-1, 1)


def generate_uncertainty_pmf(
    control_input: np.ndarray, state: np.ndarray, noise_params: List[List[float]], S: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates Probability Mass Function (PMF) and samples for uncertainty.

    Args:
        control_input: Shape (m, 1) or (m,)
        state: Shape (n, 1) or (n,)
        noise_params: List of lists/arrays.
                      noise[0] = [std_px, std_py, std_vx, std_vy] (state noise stds)
                      noise[1] = [std_ux_ux, std_ux_uy, std_uy_ux, std_uy_uy] (control dependent noise)
        S: Number of samples

    Returns:
        pmf: Probability mass function values (S, 1)
        samples_u: Control noise samples (S, m, 1)
        samples_x: State noise samples (S, n, 1)
    """
    # Reshape inputs to (dim, 1)
    u = control_input.reshape(-1, 1)
    x = state.reshape(-1, 1)

    # Parse noise params
    # Assuming structure matches original:
    # noise[0] -> State noise stds
    # noise[1] -> Control-dependent noise params

    std_px, std_py, std_vx, std_vy = 0.0, 0.0, 0.0, 0.0
    if len(noise_params[0]) >= 2:
        std_px, std_py = noise_params[0][:2]
    if len(noise_params[0]) >= 4:
        std_vx, std_vy = noise_params[0][2:4]

    std_ux_ux, std_ux_uy, std_uy_ux, std_uy_uy = 0.0, 0.0, 0.0, 0.0
    if len(noise_params[1]) >= 4:
        std_ux_ux, std_ux_uy, std_uy_ux, std_uy_uy = noise_params[1][:4]

    # Calculate control-dependent noise std dev
    # std_ux = sqrt( Coeff_xx * ux^2 + Coeff_xy * uy^2 )
    std_ux = np.sqrt(std_ux_ux * (u[0, 0] ** 2) + std_ux_uy * (u[1, 0] ** 2))
    std_uy = np.sqrt(std_uy_ux * (u[0, 0] ** 2) + std_uy_uy * (u[1, 0] ** 2))

    # Sample Control Noise
    # samples_u shape: (S, 2, 1)
    # Note: Using np.random directly here. Ideally should use a passed RNG or cbfkit's randomness,
    # but for now maintaining parity with original logic which used a class-level RNG.
    # We'll use global np.random for simplicity in this utility, or user can seed globally.
    samples_u = np.random.normal(loc=[[0], [0]], scale=[[std_ux], [std_uy]], size=(S, 2, 1))

    pdf_ux = _pdf_or_ones(samples_u[:, 0], std_ux)
    pdf_uy = _pdf_or_ones(samples_u[:, 1], std_uy)

    # Sample State Noise
    if x.shape[0] == 2:
        # 2D state
        if std_px <= 1e-5 and std_py <= 1e-5:
            std_px = 0.001
            std_py = 0.001
        samples_x = np.random.normal(loc=[[0], [0]], scale=[[std_px], [std_py]], size=(S, 2, 1))
        pdf_px = _pdf_or_ones(samples_x[:, 0], std_px)
        pdf_py = _pdf_or_ones(samples_x[:, 1], std_py)
        joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py

    else:
        # 4D state
        samples_x = np.random.normal(
            loc=[[0], [0], [0], [0]], scale=[[std_px], [std_py], [std_vx], [std_vy]], size=(S, 4, 1)
        )
        pdf_px = _pdf_or_ones(samples_x[:, 0], std_px)
        pdf_py = _pdf_or_ones(samples_x[:, 1], std_py)
        pdf_vx = _pdf_or_ones(samples_x[:, 2], std_vx)
        pdf_vy = _pdf_or_ones(samples_x[:, 3], std_vy)

        joint_pdf = pdf_ux * pdf_uy * pdf_px * pdf_py * pdf_vx * pdf_vy

    # Normalize PMF
    pmf = joint_pdf / np.sum(joint_pdf)

    return pmf, samples_u, samples_x
