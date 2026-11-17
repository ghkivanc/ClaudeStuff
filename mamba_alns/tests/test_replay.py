import numpy as np
import pytest

from mamba_alns.cvrp import initial_solution, random_instance
from mamba_alns.replay import Episode, ReplayBuffer, n_step_return, stack_windows


@pytest.fixture
def episode():
    inst = random_instance(8, 20, np.random.default_rng(0))
    routes = initial_solution(inst)
    ep = Episode(inst, routes)
    for t in range(6):                      # actions 10..15, rewards 0.0..0.5
        ep.add_step(10 + t, t / 10, routes)
    return ep


def test_episode_lengths(episode):
    assert len(episode) == 6
    assert len(episode.edges) == 7          # s_0 .. s_6


def test_window_at_start_is_padded(episode):
    graphs, prev_a, prev_r = episode.window(0, 4)
    assert len(graphs) == 4
    assert prev_a == [-1, -1, -1, -1]
    assert prev_r == [0.0, 0.0, 0.0, 0.0]


def test_window_middle(episode):
    # steps 3, 4, 5 -> previous actions a_2, a_3, a_4
    graphs, prev_a, prev_r = episode.window(5, 3)
    assert prev_a == [12, 13, 14]
    assert prev_r == pytest.approx([0.2, 0.3, 0.4])
    assert graphs[-1][1] is episode.edges[5]


def test_window_at_final_state(episode):
    _, prev_a, _ = episode.window(6, 2)
    assert prev_a == [14, 15]


def test_stack_windows_shapes(episode):
    graphs, prev_a, prev_r = stack_windows([episode.window(2, 4), episode.window(5, 4)])
    assert len(graphs) == 8
    assert prev_a.shape == (2, 4) and prev_r.shape == (2, 4)


def test_n_step_return():
    rewards = [1.0, 2.0, 3.0, 4.0]
    assert n_step_return(rewards, 0, 2, 0.5) == (pytest.approx(1 + 0.5 * 2), True)
    assert n_step_return(rewards, 1, 2, 0.5) == (pytest.approx(2 + 0.5 * 3), True)
    # s_{t+n} = s_4 is the final state -> no bootstrap
    assert n_step_return(rewards, 2, 2, 0.5) == (pytest.approx(3 + 0.5 * 4), False)
    # cut short at the end
    assert n_step_return(rewards, 3, 2, 0.5) == (pytest.approx(4.0), False)


def test_n_step_one_is_plain_q_learning():
    rewards = [1.0, 2.0, 3.0]
    assert n_step_return(rewards, 0, 1, 0.9) == (pytest.approx(1.0), True)


def test_buffer_sample(episode):
    buf = ReplayBuffer(max_episodes=2)
    buf.add(episode)
    assert buf.n_transitions() == 6
    for ep, t in buf.sample(20):
        assert ep is episode and 0 <= t < 6
