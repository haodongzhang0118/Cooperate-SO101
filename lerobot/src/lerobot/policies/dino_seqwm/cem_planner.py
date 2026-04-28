"""Simplified CEM planner for LeRobot integration.

Adapted from dino_wm/planning/cem.py — removes dependencies on
BasePlanner, preprocessor, evaluator, wandb, etc.
"""

import torch
import torch.nn as nn
from einops import repeat


class CEMPlanner:
    """Cross-Entropy Method planner for goal-conditioned visual planning.

    Samples action sequences, rolls them out through the world model,
    evaluates against a goal latent, and iteratively refines via elites.

    Args:
        horizon: Planning horizon (number of future action steps).
        topk: Number of elite trajectories to select each iteration.
        num_samples: Number of action sequences to sample per iteration.
        opt_steps: Number of CEM optimization iterations.
        action_dim: Dimensionality of the action space.
        var_scale: Initial standard deviation for action sampling.
    """

    def __init__(
        self,
        horizon: int = 5,
        topk: int = 10,
        num_samples: int = 100,
        opt_steps: int = 30,
        action_dim: int = 6,
        var_scale: float = 1.0,
    ):
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.opt_steps = opt_steps
        self.action_dim = action_dim
        self.var_scale = var_scale

    @torch.no_grad()
    def plan(self, rollout_fn, objective_fn, z_obs_goal, device, actions=None):
        """Plan an action sequence via CEM.

        Args:
            rollout_fn: Callable(action_batch) -> z_obses dict.
                action_batch: (num_samples, horizon, action_dim)
                Returns: {"visual": (num_samples, T, ...), "proprio": (num_samples, T, ...)}
            objective_fn: Callable(z_obs_pred, z_obs_tgt) -> loss (num_samples,)
            z_obs_goal: Dict with "visual" and "proprio" goal latents, each (1, ...).
            device: torch device.
            actions: Optional warm-start actions (1, T, action_dim).

        Returns:
            best_actions: (horizon, action_dim) — the optimized action sequence.
        """
        # Initialize mu and sigma
        mu = torch.zeros(self.horizon, self.action_dim, device=device)
        sigma = self.var_scale * torch.ones(self.horizon, self.action_dim, device=device)

        if actions is not None:
            t = actions.shape[0]
            mu[:t] = actions

        for _ in range(self.opt_steps):
            # Sample action sequences: (num_samples, horizon, action_dim)
            noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=device)
            action_samples = mu.unsqueeze(0) + noise * sigma.unsqueeze(0)
            action_samples[0] = mu  # Include the current mean

            # Rollout through world model
            z_obses = rollout_fn(action_samples)

            # Expand goal to match num_samples
            z_goal_expanded = {
                k: repeat(v, "1 ... -> n ...", n=self.num_samples)
                for k, v in z_obs_goal.items()
            }

            # Evaluate
            loss = objective_fn(z_obses, z_goal_expanded)

            # Select elites
            topk_idx = torch.argsort(loss)[: self.topk]
            elites = action_samples[topk_idx]

            # Update distribution
            mu = elites.mean(dim=0)
            sigma = elites.std(dim=0)

        return mu
