import torch

from plan_rl.leduc import CALL, RAISE
from plan_rl.model import CHANCE, MY, OPP, PAD, WorldModel, history_events, to_event


def small_model(n_types=2):
    torch.manual_seed(0)
    return WorldModel(n_types, d_model=16, mamba_layers=1, hidden=32)


def test_events_from_each_seat():
    rounds = ((CALL, RAISE, CALL), (CALL,))
    assert history_events(0, 2, rounds) == [(MY, CALL), (OPP, RAISE), (MY, CALL),
                                            (CHANCE, 2), (MY, CALL)]
    assert history_events(1, 2, rounds)[0] == (OPP, CALL)
    assert [to_event(MY, 2), to_event(OPP, 0), to_event(CHANCE, 1)] == [2, 3, 7]


def test_represent_shapes():
    model = small_model()
    assert model.represent_hypotheses(0, 1, []).shape == (6, 16)
    assert model.represent_hypotheses(1, 2, [3, 1, 6]).shape == (6, 16)


def test_padding_does_not_change_latent():
    model = small_model()

    def zeros(b):  # seat, my card, opponent card, opponent type
        return [torch.zeros(b, dtype=torch.long)] * 4

    alone = model.represent(*zeros(1), torch.tensor([[4, 1]]), torch.tensor([2]))
    batched = model.represent(*zeros(2), torch.tensor([[4, 1, PAD, PAD], [4, 1, 6, 1]]),
                              torch.tensor([2, 4]))
    assert torch.allclose(alone[0], batched[0], atol=1e-5)


def test_dynamics_and_heads():
    model = small_model()
    latent = model.represent_hypotheses(0, 0, [])
    nxt, reward = model.dynamics(latent, torch.full((6,), 1))
    pred = model.predict(nxt)
    assert nxt.shape == (6, 16) and reward.shape == (6,)
    assert pred.kind.shape == (6, 4) and pred.my_policy.shape == (6, 3)
    assert pred.value.shape == (6,)
