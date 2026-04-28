"""Objective functions for CEM planning, copied from dino_wm/planning/objectives.py."""

import numpy as np
import torch
import torch.nn as nn


def create_objective_fn(alpha, base=1.0, mode="last"):
    """Create an objective function for CEM planning.

    Args:
        alpha: Weight for proprioceptive loss relative to visual loss.
        base: Exponential base for weighting frames in "all" mode.
        mode: "last" — evaluate only last predicted frame.
              "all" — evaluate all frames with exponential weighting.

    Returns:
        Callable: objective_fn(z_obs_pred, z_obs_tgt) -> loss (B,)
    """
    metric = nn.MSELoss(reduction="none")

    def objective_fn_last(z_obs_pred, z_obs_tgt):
        loss_visual = metric(
            z_obs_pred["visual"][:, -1:], z_obs_tgt["visual"]
        ).mean(dim=tuple(range(1, z_obs_pred["visual"].ndim)))

        loss_proprio = metric(
            z_obs_pred["proprio"][:, -1:], z_obs_tgt["proprio"]
        ).mean(dim=tuple(range(1, z_obs_pred["proprio"].ndim)))

        return loss_visual + alpha * loss_proprio

    def objective_fn_all(z_obs_pred, z_obs_tgt):
        coeffs = np.array(
            [base**i for i in range(z_obs_pred["visual"].shape[1])], dtype=np.float32
        )
        coeffs = torch.tensor(coeffs / np.sum(coeffs)).to(z_obs_pred["visual"].device)

        loss_visual = metric(z_obs_pred["visual"], z_obs_tgt["visual"]).mean(
            dim=tuple(range(2, z_obs_pred["visual"].ndim))
        )
        loss_proprio = metric(z_obs_pred["proprio"], z_obs_tgt["proprio"]).mean(
            dim=tuple(range(2, z_obs_pred["proprio"].ndim))
        )
        loss_visual = (loss_visual * coeffs).mean(dim=1)
        loss_proprio = (loss_proprio * coeffs).mean(dim=1)
        return loss_visual + alpha * loss_proprio

    if mode == "last":
        return objective_fn_last
    elif mode == "all":
        return objective_fn_all
    else:
        raise NotImplementedError(f"Unknown objective mode: {mode}")
