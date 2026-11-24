# Diffusion-sampled policy weights

Policy weights are not trained directly. A conditional diffusion model
**samples** them, conditioned on an encoding of the last K (s, a, r, s')
transitions. The sampler improves through a reward-weighted denoising loss
derived from a KL-regularised improvement step (the same derivation as REPS
and AWR). β is tuned every iteration to hit a KL target.

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
| `diversity` | mean pairwise distance between canonical samples; a steady fall to ~0 means collapse |
| `cond_gap` | loss with shuffled histories minus loss with true ones; ~0 means h is ignored |

At the end of each episode it also prints the mean episode return over all N pendulums.

## How it works

### One iteration

1. Snapshot each environment's history Hᵢ: the K tuples before the new policy
   takes control.
2. Encode it, hᵢ = e_ψ(Hᵢ), and sample θᵢ with DDIM, conditioned on hᵢ.
3. Run θᵢ for one segment of L = `segment_multiple` · K steps, or fewer if the
   episode ends first.
4. Return: Rᵢ = Σ γᵗ rₜ + γᴸ V(H at segment end). The last s′ of that history
   is the state the next policy starts from.
5. Advantage Aᵢ = Rᵢ − V(Hᵢ). β is chosen by bisection so that
   w = softmax(A / β) is exactly ε_KL from uniform.
6. G gradient steps. Each one picks indices with probability wᵢ, applies the
   denoising loss to the canonical zᵢ, and adds the value loss on a uniform
   minibatch. The value loss trains the encoder as well.
7. Discard the batch (on-policy). Episodes carry on across iterations, and
   older policies only ever appear as conditioning.

All N pendulums share one step counter, so their episodes end together. When
they do, every pendulum gets a new mass and length and the histories are
cleared. The first segment of an episode is conditioned on the null embedding.

### Bootstrap

Before iteration 1, the denoiser is pretrained on canonicalised random
initialisations, with h set to the null embedding, so q₀ is "a reasonable
random policy" rather than noise. The weight normaliser's per-coordinate
statistics come from this set.

Both the null embedding and the denoiser's FiLM layers start at zero. The
bootstrap therefore leaves the weights that read h at exactly zero, and a
real h starts with no effect.

### Canonicalization (two passes)

1. **Scale**, every hidden layer from input to output: give each unit's incoming
   row [W_in[j, :], b_j] unit norm, and move the norm into its outgoing column.
2. **Permutation**, every hidden layer: sort units by ‖W_out[:, j]‖ descending,
   with b_j breaking ties.

The design document interleaves the two steps layer by layer. In that order a
layer's sort key still contains the next layer's arbitrary per-unit scales,
so equivalent networks can sort differently. Doing all the scale removal first
fixes this. `test_canonicalizer_removes_symmetry` scales every hidden layer,
so it would catch the old order.

### The policy

The policy is a ReLU MLP with layers 3 → 32 → 32 → 2 (1,250 weights). Its two
outputs are the action mean and log-std. The action is sampled from that
Gaussian, then clipped to the ±2 torque limit.

### Differences from the design document

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
| 1–3 canonicalizer preserves function, idempotent, removes symmetry | `tests/test_canonicalize.py` |
| 4 β tuner hits its target | `tests/test_beta.py` |
| 5 loss reduces to standard DDPM | `tests/test_diffusion.py` |
| 6 encoder masking | `tests/test_encoder.py` |
| 7 no leakage | `tests/test_train.py` |
| smoke 1 diffusion sanity (2-D Gaussian mixture) | `tests/test_diffusion.py` |
| smoke 2 environment-free improvement | `tests/test_train.py` |

There are also tests for the pendulum dynamics, for segments stopping at the
end of an episode, and for one full training iteration running end to end.

## Layout

```
diffusion_policy/
  config.py        all hyperparameters
  pendulum.py      N pendulums with per-episode mass and length
  policy.py        flat-weight MLP policy, batched over N weight vectors
  canonicalize.py  two-pass symmetry removal
  history.py       rolling window of the last K (s, a, r, s') tuples
  encoder.py       history encoder e_psi (MLP embed, positions, transformer, pooling)
  denoiser.py      FiLM-conditioned MLP denoiser eps_phi
  diffusion.py     noise schedule, denoising loss, DDIM sampling
  beta.py          beta by bisection on the KL target
  model.py         weight normaliser and the module holding all trainable parts
  train.py         bootstrap, collect, refit, CLI
  tests/
```
