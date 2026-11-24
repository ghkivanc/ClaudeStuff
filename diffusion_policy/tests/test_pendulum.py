import math

import torch

from diffusion_policy.pendulum import Pendulums


def test_reset_draws_parameters_in_range():
    torch.manual_seed(0)
    env = Pendulums(256, (0.5, 2.0), (0.5, 1.5), max_steps=200)
    assert (env.mass >= 0.5).all() and (env.mass <= 2.0).all()
    assert (env.length >= 0.5).all() and (env.length <= 1.5).all()
    assert env.mass.std() > 0.1 and env.length.std() > 0.1


def test_upright_at_rest_stays_with_zero_reward():
    env = Pendulums(3, (0.5, 2.0), (0.5, 1.5), max_steps=200)
    env.angle.zero_()
    env.angular_velocity.zero_()
    obs, reward = env.step(torch.zeros(3, 1))
    torch.testing.assert_close(obs, torch.tensor([[1.0, 0.0, 0.0]] * 3))
    torch.testing.assert_close(reward, torch.zeros(3))


def test_torque_effect_depends_on_mass_and_length():
    env = Pendulums(2, (0.5, 2.0), (0.5, 1.5), max_steps=200)
    env.mass = torch.tensor([0.5, 2.0])
    env.length = torch.tensor([0.5, 1.5])
    env.angle = torch.full((2,), math.pi)  # hanging down: gravity term is ~0
    env.angular_velocity.zero_()
    obs, _ = env.step(torch.full((2, 1), 2.0))
    assert obs[0, 2] > 5 * obs[1, 2]  # light, short pendulum accelerates far more


def test_episode_ends_after_max_steps():
    env = Pendulums(2, (0.5, 2.0), (0.5, 1.5), max_steps=5)
    for _ in range(5):
        assert not env.done()
        env.step(torch.zeros(2, 1))
    assert env.done()
