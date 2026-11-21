# Plan-based self-play RL for Leduc poker

Agents choose moves by **making plans**, not by reacting to returns of single
moves. At each decision an agent:

1. samples many **plans**, which are sequences of (s, a, r) tuples rolled out
   in its own learned world model. A plan contains its own moves, the moves it
   expects from the opponent, and the public card;
2. merges the plans into a **tree**, where plans that start the same way share
   branches;
3. **combines the plan values** using its **belief** over the opponent's
   hidden card;
4. **commits** to the plan and follows it, sampling each of its moves from the
   plan's policy, until the opponent does something the plan did not expect.
   Then it replans.

Training is AlphaZero-style self-play: the current model plays both seats.
Each seat is a separate instance with its own tree and its own belief, as in a
real game. The game is Leduc hold'em, a small poker game
with hidden information, so this is a genuine Bayesian game.

## How it works

### The game
- **Deck:** J, J, Q, Q, K, K. Each player antes 1 chip and gets one private card.
- **Round 1:** betting with raises of 2.
- **Round 2:** the public card is dealt, then betting with raises of 4. At most
  2 raises per round, and player 0 acts first in each round.
- **Showdown:** pairing the public card wins; otherwise the higher card wins.
  Equal ranks split the pot.
- **Actions:** fold, call/check, raise.
- **Reward:** chips won or lost, paid only at the end of the hand.

### The belief
- **Hypotheses:** the opponent's card, J, Q or K.
- **Prior:** from the deck after removing my card (if I hold J: J 1/5,
  Q 2/5, K 2/5).
- **Public card:** exact Bayes from the deck.
- **Opponent action:** Bayes, using the world model's opponent head as the
  likelihood.

This last update is **approximate**. The opponent also plans, and their choice
depends on their own belief about my card. The opponent head sees the public
history and their card, which is everything that belief depends
on (the opponent is always the current model). So it can learn the opponent's
behaviour without nesting beliefs about beliefs explicitly.

### The world model
The model is MuZero-style, and planning never uses the rules.

| Part | Input → output |
|---|---|
| representation (Mamba) | my seat and card, one possible opponent card, event history → latent |
| dynamics (MLP) | latent, event → next latent, reward for me |
| heads (MLP) | latent → node kind (my turn / opponent's turn / chance / end), my policy, opponent policy, public-card probabilities, legal actions, value |

### The plan tree
- **Nodes:** each node is a position in the public sequence of events. It holds
  one latent per hypothesis and the belief given the path to it.
- **Sampling plans:** each step of a plan comes from a different source:
  - my moves: my policy prior, restricted to the legal actions;
  - opponent moves: the opponent head, averaged over the belief;
  - the public card: the chance head.
- **Belief inside the tree:** it is updated by Bayes along every opponent and
  chance branch.
- **Backup:** values are kept per hypothesis.
  - **My node:** each action's value `q(a)` is its belief-weighted value. My
    policy there is `softmax(q / policy_temperature)`, the **same** for every
    hypothesis: I can't tell the opponent's cards apart, so I can't choose
    differently for each. This avoids strategy fusion. The node's value is
    the expectation under that policy, which matches how the agent actually
    plays.
  - **Opponent or chance node:** the expectation over its outcomes, separately
    for each hypothesis.
  - **Unexpanded leaf:** the value head.
  - **End of hand:** 0, because rewards sit on the edges.

### Commit until surprised
- **The plan:** the tree's policy at every one of my nodes, so it is a
  contingent plan, not a single line of play.
- **Following it:** at each of its turns the agent samples its action from
  the plan's policy there, limited to the real legal actions. Sampling gives
  the mixed strategies poker needs. The real belief updates after every
  event.
- **When it replans:** if the opponent's action or the public card isn't in the
  tree, or had probability below `surprise_threshold` there.

### Learning
1. **Self-play:** each hand, the current model plays itself, one
   `PlanningAgent` per seat. Each hand gives two records, one per seat.
2. **Records:** after a hand both cards are known, so every head has a target:
   - node kind and legal actions: from the rules;
   - my policy: the policy I actually sampled from at each of my decisions;
   - opponent policy: the action they actually took;
   - chance: the card actually dealt;
   - reward and value: the chips I won.
3. **Training:** MuZero-style. Represent a random prefix of the hand using the
   true opponent card, then unroll the dynamics over the rest of the hand.

## Setup

Requires Python 3.10+.

```bash
git clone <this repo> && cd ClaudeStuff
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1. PyTorch, CPU build (planning runs on the CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else (numpy, pytest)
pip install -r requirements.txt
```

Everything runs on the CPU. The networks are tiny and planning makes many
small calls, which a GPU would not speed up.

## Run the tests

From the repository root:

```bash
pytest plan_rl/tests
```

| File | Checks |
|---|---|
| `test_leduc.py` | legal actions, round ends, bet sizes, fold and showdown payoffs, replay order |
| `test_cfr.py` | deal probabilities, exact EV on a symmetric game, CFR policy validity, CFR beats random |
| `test_belief.py` | deck prior, exact public-card update, action update |
| `test_model.py` | event numbering, shapes, right padding does not change latents |
| `test_plan_tree.py` | root expands the legal actions, root policy = softmax of values, plans share prefixes, belief-weighted q, my value = expectation under the policy, Bayes on opponent branches |
| `test_agent.py` | plays legal hands, impossible cards get belief 0, follows a plan without replanning, plan policy limited to real legal actions, surprise drops the plan |
| `test_train.py` | records built from a hand, loss decreases on a fixed batch, training smoke run |
| `test_mamba.py` | Mamba shape, causality, gradients |

## A whole run

From the repository root:

```bash
python -m plan_rl.train
```

1. **Self-play training** for 5000 hands. Every 200 hands it prints a line like:
   ```
   hand   1200  seat-0 payoff/hand -0.080  replans/player/hand 1.85  loss 3.412
   ```
   - `seat-0 payoff/hand`: chips won by whoever acts first. It is the same
     model in both seats, so this only shows the first-mover (dis)advantage.
   - `replans/player/hand`: how often plans were dropped. Lower means it
     predicts its opponent better.
   - `loss`: the world-model training loss.
2. **Saves** the world model to `world_model.pt`.
3. **Evaluates.** It runs CFR (1000 iterations), prints CFR's exact result
   against random as a reference, then plays the agent for 1000 hands each
   against:
   - **random:** should be clearly positive;
   - **cfr:** a near-equilibrium strategy. About 0 chips/hand is the best you
     can expect on average, and clearly negative means the agent is
     exploitable.

To re-evaluate a saved model: `python -m plan_rl.evaluate`.

### Play against it yourself

```bash
python -m plan_rl.play            # against the trained bot (loads world_model.pt)
python -m plan_rl.play --cfr      # against the CFR strategy instead
```

You and the bot swap seats every hand. Type `f`, `c` or `r` to fold,
check/call or bet/raise, and `q` to stop. After each hand it shows the bot's
card and your running score.

Rough CPU runtimes (estimated, not measured): about 0.5–1 s per self-play
hand, because both players plan. So the default 5000 hands take **~1–1.5 h**.
CFR takes a few minutes and evaluation ~15–30 min. For a quick trial:

```python
from plan_rl.train import Config, train
from plan_rl.evaluate import evaluate_all

cfg = Config(hands=400, n_plans=16, eval_hands=200, cfr_iterations=200)
evaluate_all(train(cfg), cfg)
```

## Configuration

All settings are fields of `Config` in `plan_rl/train.py`.

| Field | Default | Meaning |
|---|---|---|
| `n_plans` | 32 | plans sampled per decision |
| `max_depth` | 10 | maximum events per plan (a hand has at most 9) |
| `surprise_threshold` | 0.05 | replan when the observed event had lower probability in the plan |
| `policy_temperature` | 1.0 | softmax temperature (in chips) of the plan's policy, used both for acting and as the training target |
| `d_model`, `mamba_layers` | 64, 2 | world-model size |
| `hands` | 5000 | self-play hands |
| `train_every`, `updates_per_phase` | 20, 10 | gradient steps per 20 hands |
| `batch_size`, `buffer_records`, `lr` | 64, 10000, 3e-4 | learning (two records per hand) |
| `log_every` | 200 | hands between log lines |
| `eval_hands`, `cfr_iterations` | 1000, 1000 | evaluation |
| `seed` | 0 | seeds Python, numpy and torch |

## Project layout

```
plan_rl/
  leduc.py      game rules, play_hand
  cfr.py        CFR baseline, exact expected value of two fixed policies
  belief.py     belief over the opponent's card: prior, Bayes updates
  mamba.py      Mamba block (selective scan written as a loop)
  model.py      world model: representation, dynamics, heads; event numbering
  plan_tree.py  plans, tree nodes, backup, policy target
  agent.py      planning agent: belief, commit until surprised, replanning
  train.py      Config, records, loss, self-play loop
  evaluate.py   matches against random and CFR
  play.py       play against the bot in the terminal
  tests/
```

## Notes and limitations

- **Temperature.** `policy_temperature` is in chips. Too low and play becomes
  nearly deterministic, which is exploitable in poker. Too high and it ignores
  the plan values.
- **Approximate opponent likelihood.** It comes from a learned head (see the
  belief section). Early in training the belief is only as good as that head.
- **Model errors.** Planning trusts the learned model, including its node
  kinds and legal actions. At the real root, the true legal actions are used.
- **Self-play only.** The agent only ever trains against the current model,
  so its opponent head models itself. Against a different opponent (e.g.
  CFR), the opponent head and therefore the belief updates are misspecified.
