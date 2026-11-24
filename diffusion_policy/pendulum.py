"""N pendulum swing-up environments stepped together, each with its own mass and length.

Dynamics, observation and reward follow the standard Pendulum task: the
pendulum is upright at angle 0, the observation is (cos angle, sin angle,
angular velocity), and the reward penalises angle, speed and torque.
"""

import math

import torch

OBS_DIM = 3
ACT_DIM = 1
MAX_TORQUE = 2.0
MAX_SPEED = 8.0
GRAVITY = 10.0
DT = 0.05


def angle_normalize(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class Pendulums:
    def __init__(self, num_envs, mass_range, length_range, max_steps, device="cpu"):
        self.num_envs = num_envs
        self.mass_range = mass_range
        self.length_range = length_range
        self.max_steps = max_steps
        self.device = device
        self.reset()

    def _uniform(self, low, high):
        return low + (high - low) * torch.rand(self.num_envs, device=self.device)

    def reset(self):
        """Start a new episode in every environment, with new masses and lengths."""
        self.mass = self._uniform(*self.mass_range)
        self.length = self._uniform(*self.length_range)
        self.angle = self._uniform(-math.pi, math.pi)
        self.angular_velocity = self._uniform(-1.0, 1.0)
        self.t = 0

    def observe(self):
        return torch.stack([self.angle.cos(), self.angle.sin(), self.angular_velocity], dim=-1)

    def step(self, action):
        """action: (N, 1). Returns (next observation (N, 3), reward (N,))."""
        torque = action.squeeze(-1).clamp(-MAX_TORQUE, MAX_TORQUE)
        cost = angle_normalize(self.angle) ** 2 + 0.1 * self.angular_velocity**2 + 0.001 * torque**2

        acceleration = (
            3 * GRAVITY / (2 * self.length) * self.angle.sin()
            + 3.0 / (self.mass * self.length**2) * torque
        )
        self.angular_velocity = (self.angular_velocity + acceleration * DT).clamp(-MAX_SPEED, MAX_SPEED)
        self.angle = self.angle + self.angular_velocity * DT
        self.t += 1
        return self.observe(), -cost

    def done(self):
        """All environments share one step counter, so episodes end together."""
        return self.t >= self.max_steps
