# Diffusion-sampled policy weights

Policy weights are not trained directly. A conditional diffusion model
**samples** them, conditioned on an encoding of the last K (s, a, r, s')
transitions. In effect the policy is a Bayesian neural network whose weight
distribution is a diffusion model instead of a Gaussian. The sampler improves
through a reward-weighted denoising loss derived from a KL-regularised
improvement step (the same derivation as REPS and AWR). β is tuned every
iteration to hit a KL target.

The test environment is a batch of **pendulums with different masses and
lengths**. Each episode draws new values and hides them from the policy. A
heavy, long pendulum needs a different swing-up policy than a light, short
one, so the sampler has to read the history to tell which pendulum it is on.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m diffusion_policy.train                      # all defaults, CPU
python -m diffusion_policy.train --iterations 200 --device cuda
python -m diffusion_policy.train --segment_multiple 1 --eps_kl 0.5
```

Every scalar field in `diffusion_policy/config.py` is also a `--flag`. Tuples
(hidden sizes, mass and length ranges) are changed in `config.py`. The trained
model is saved to `sampler.pt`.

Each iteration prints:

| Column | Meaning |
|---|---|
| `reward/step` | mean raw reward per step over the segment (0 is the best possible) |
| `beta`, `kl` | the tuned β and the KL of the batch weights from uniform (should equal `eps_kl`) |
| `denoise`, `value` | the last minibatch losses of the refit |
| `diversity` | mean pairwise distance between sampled z; about √(2P) ≈ 50 for healthy unit-scale samples, a steady fall to ~0 means collapse |
| `cond_gap` | loss with shuffled histories minus loss with true ones; ~0 means h is ignored |

At the end of each episode it also prints the mean episode return over all N pendulums.

## How it works

### One iteration

N pendulums run in parallel. They share one sampler, but each pendulum gets
its own policy weights θᵢ, drawn for its own history.

1. Snapshot each environment's history Hᵢ: the K tuples before the new policy
   takes control.
2. Encode it, hᵢ = e_ψ(Hᵢ), and sample zᵢ with DDIM, conditioned on hᵢ. The
   policy weights are θᵢ = unnormalize(zᵢ).
3. Run θᵢ for one segment of L = `segment_multiple` · K steps, or fewer if the
   episode ends first.
4. Return: Rᵢ = Σ γᵗ rₜ + γᴸ V(H at segment end). The last s′ of that history
   is the state the next policy starts from.
5. Advantage Aᵢ = Rᵢ − V(Hᵢ). β is chosen by bisection so that
   w = softmax(A / β) is exactly ε_KL from uniform.
6. G gradient steps. Each one picks indices with probability wᵢ, applies the
   denoising loss to zᵢ exactly as sampled, and adds the value loss on a
   uniform minibatch. The value loss trains the encoder as well.
7. Discard the batch (on-policy). Episodes carry on across iterations, and
   older policies only ever appear as conditioning.

All N pendulums share one step counter, so their episodes end together. When
they do, every pendulum gets a new mass and length and the histories are
cleared. The first segment of an episode is conditioned on the null embedding.

### The denoiser: a U-Net over the weights as an image

Every layer's W and b are vectorised and concatenated into θ (1,250 numbers
for the default policy), standardised per coordinate, zero-padded and reshaped
into a 36 × 36 image. The layout is arbitrary. Two fixed coordinate channels
(CoordConv: each cell's row and column in [−1, 1]) are added to the image, so
the network knows which weight each cell holds. A U-Net then predicts the
noise:
- it has two halvings (36 → 18 → 9) with skip connections;
- each resolution has a residual block with FiLM on (t, h);
- self-attention at the 9 × 9 level lets every part of the image reach every
  other part, because related weights can land far apart in the arbitrary
  layout;
- the output is cropped back to the flat vector.

Padded cells are only ever zeros on the input side, and the loss only sees the
real 1,250 outputs.

### No canonicalization

Sampled weights are stored exactly as sampled. An earlier version
re-canonicalized every sample while preserving its function. That pushed the
sampler's per-unit scale noise into the output layer, where it compounded
every iteration (the weights blew up about 170× within 60 iterations). The
sampler is only ever trained on its own samples, so it rarely sees the same
network in two unit orders. The one exception is the bootstrap set of random
initialisations, where every weight is independent and every ordering looks
alike.

### Bootstrap

Before iteration 1, the denoiser is pretrained on random initialisations,
with h set to the null embedding, so q₀ is "a reasonable random policy"
rather than noise. The weight normaliser's per-coordinate statistics come
from this set.

Both the null embedding and the FiLM layers start at zero. The bootstrap
therefore leaves the weights that read h at exactly zero, and a real h starts
with no effect.

### The policy

The policy is a ReLU MLP with layers 3 → 32 → 32 → 2. Its two outputs are the
action mean and log-std. The action is sampled from that Gaussian, then
clipped to the ±2 torque limit.

### Differences from the design document

- The denoiser is a U-Net over the weights laid out as an image, not an MLP.
- There is no canonicalization.
- Tuples are (s, a, r, s′), not (s, a, r), so the history contains the state
  the policy starts in.
- V(h) is a third trainable component, used for bootstrapped returns and the
  advantage baseline.
- The diffusion schedule is the DDPM linear schedule rescaled to T steps.
  The cosine schedule drops too steeply in its last steps for deterministic
  DDIM sampling from pure noise.

## Tests

```bash
pytest                    # everything
pytest -m "not smoke"     # unit tests only (fast)
```

| Design doc test | Where |
|---|---|
| 4 β tuner hits its target | `tests/test_beta.py` |
| 5 loss reduces to standard DDPM | `tests/test_diffusion.py` |
| 6 encoder masking | `tests/test_encoder.py` |
| 7 no leakage | `tests/test_train.py` |
| smoke 1 diffusion sanity (2-D Gaussian mixture) | `tests/test_diffusion.py` |
| smoke 2 environment-free improvement | `tests/test_train.py` |

Tests 1–3 tested the canonicalizer, which no longer exists.

There are also tests for:
- the pendulum dynamics;
- the U-Net's shapes, and h having no effect at initialisation;
- the bootstrap leaving the h-weights at zero;
- the stored z being the sampled z;
- segments stopping at the end of an episode;
- one full training iteration running end to end.

## Layout

```
diffusion_policy/
  config.py        all hyperparameters
  pendulum.py      N pendulums with per-episode mass and length
  policy.py        flat-weight MLP policy, batched over N weight vectors
  history.py       rolling window of the last K (s, a, r, s') tuples
  encoder.py       history encoder e_psi (MLP embed, positions, transformer, pooling)
  denoiser.py      U-Net denoiser eps_phi over the weights as an image, FiLM on (t, h)
  diffusion.py     noise schedule, denoising loss, DDIM sampling
  beta.py          beta by bisection on the KL target
  model.py         weight normaliser and the module holding all trainable parts
  train.py         bootstrap, collect, refit, CLI
  tests/
```
