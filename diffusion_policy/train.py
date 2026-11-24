"""Bootstrap the sampler, then run improvement iterations on the pendulums.

One iteration:
  1. encode each environment's history H_i (the K tuples before theta_i acts);
  2. sample theta_i from the diffusion model conditioned on h_i;
  3. run theta_i for one segment (L steps, or fewer if the episode ends);
  4. R_i = discounted segment reward + gamma^len * V(history at segment end);
  5. A_i = R_i - V(H_i); beta from the KL target; w = softmax(A / beta);
  6. G gradient steps: weighted denoising loss on the sampled z_i, plus value loss.

Run: python -m diffusion_policy.train [--iterations 200 --device cuda ...]
"""

import argparse
from dataclasses import asdict, fields

import torch

from .beta import improvement_weights, kl_to_uniform
from .config import Config
from .diffusion import Diffusion
from .history import History, tuple_dim
from .model import SamplerModel
from .pendulum import ACT_DIM, OBS_DIM, Pendulums
from .policy import PolicySpec, random_init, sample_action


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        self.spec = PolicySpec(OBS_DIM, cfg.hidden_sizes, ACT_DIM)
        dim = tuple_dim(OBS_DIM, ACT_DIM)
        self.env = Pendulums(cfg.num_envs, cfg.mass_range, cfg.length_range, cfg.max_steps, self.device)
        self.history = History(cfg.num_envs, cfg.K, dim, self.device)
        self.diffusion = Diffusion(cfg.diffusion_steps, self.device)
        self.model = SamplerModel(self.spec.num_params, dim, cfg).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.episode_return = torch.zeros(cfg.num_envs, device=self.device)

    def _step(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def bootstrap(self):
        """Fit the normaliser and pretrain the denoiser on random initialisations."""
        cfg = self.cfg
        theta = random_init(self.spec, cfg.bootstrap_size, self.device)
        self.model.normalizer.fit(theta)
        z = self.model.normalizer.normalize(theta)

        for step in range(1, cfg.bootstrap_steps + 1):
            idx = torch.randint(len(z), (cfg.minibatch,), device=self.device)
            h = self.model.encoder.null.expand(cfg.minibatch, -1)
            loss = self.diffusion.per_sample_loss(self.model.denoiser, z[idx], h).mean()
            self._step(loss)
            if step % 500 == 0:
                print(f"bootstrap {step:5d}  loss {loss.item():.4f}")

    @torch.no_grad()
    def sample_policies(self, h):
        """Returns the sampled standardised weights z and the policy weights theta."""
        z = self.diffusion.sample(
            self.model.denoiser, h, self.spec.num_params, self.cfg.sampling_steps, self.cfg.clip_x0
        )
        return z, self.model.normalizer.unnormalize(z)

    @torch.no_grad()
    def collect(self):
        """Run one segment in every environment. Returns the batch and some stats."""
        cfg = self.cfg
        H, H_mask = self.history.snapshot()  # taken before theta acts: no leakage
        z, theta = self.sample_policies(self.model.encoder(H, H_mask))

        segment_length = min(cfg.segment_length, cfg.max_steps - self.env.t)
        segment_return = torch.zeros(cfg.num_envs, device=self.device)
        raw_reward = torch.zeros(cfg.num_envs, device=self.device)
        discount = 1.0
        for _ in range(segment_length):
            obs = self.env.observe()
            action = sample_action(self.spec, theta, obs)
            next_obs, reward = self.env.step(action)
            raw_reward += reward
            scaled = reward * cfg.reward_scale
            self.history.push(obs, action, scaled, next_obs)
            segment_return += discount * scaled
            discount *= cfg.gamma

        # Bootstrap from the history at the end of the segment. Its last s' is the
        # state the next policy starts from. At the episode's time limit this is
        # still right: the pendulum would keep going.
        end_value = self.model.value(self.model.encoder(*self.history.snapshot()))
        returns = segment_return + discount * end_value

        batch = {
            "H": H,
            "H_mask": H_mask,
            "z": z,  # stored exactly as sampled
            "returns": returns,
        }
        stats = {"reward_per_step": (raw_reward / segment_length).mean().item()}

        self.episode_return += raw_reward
        if self.env.done():
            stats["episode_return"] = self.episode_return.mean().item()
            self.env.reset()
            self.history.clear()
            self.episode_return.zero_()
        return batch, stats

    def refit(self, batch):
        """KL-regularised improvement step: weighted denoising loss plus value loss."""
        cfg = self.cfg
        H, H_mask, z, returns = batch["H"], batch["H_mask"], batch["z"], batch["returns"]
        n = len(z)

        with torch.no_grad():
            advantages = returns - self.model.value(self.model.encoder(H, H_mask))
        weights, beta = improvement_weights(advantages, cfg.eps_kl)

        for _ in range(cfg.grad_steps):
            # Picking i with probability w_i has the same expected loss as weighting by w_i.
            idx = torch.multinomial(weights, cfg.minibatch, replacement=True)
            h = self.model.encoder(H[idx], H_mask[idx])
            denoise_loss = self.diffusion.per_sample_loss(self.model.denoiser, z[idx], h).mean()

            idx = torch.randint(n, (cfg.minibatch,), device=self.device)
            predicted = self.model.value(self.model.encoder(H[idx], H_mask[idx]))
            value_loss = ((predicted - returns[idx]) ** 2).mean()

            self._step(denoise_loss + cfg.value_coef * value_loss)

        return {
            "beta": beta,
            "kl": kl_to_uniform(weights),
            "denoise_loss": denoise_loss.item(),
            "value_loss": value_loss.item(),
            "diversity": torch.pdist(z).mean().item(),
            "conditioning_gap": self.conditioning_gap(batch),
        }

    @torch.no_grad()
    def conditioning_gap(self, batch):
        """Loss with shuffled histories minus loss with the true ones.

        Near zero means the denoiser ignores h.
        """
        H, H_mask, z = batch["H"], batch["H_mask"], batch["z"]
        n = len(z)
        t = torch.randint(1, self.diffusion.T + 1, (n,), device=self.device)
        eps = torch.randn_like(z)
        h = self.model.encoder(H, H_mask)
        shuffled = h[torch.randperm(n, device=self.device)]
        true_loss = self.diffusion.per_sample_loss(self.model.denoiser, z, h, t, eps).mean()
        shuffled_loss = self.diffusion.per_sample_loss(self.model.denoiser, z, shuffled, t, eps).mean()
        return (shuffled_loss - true_loss).item()

    def train(self):
        self.bootstrap()
        for iteration in range(1, self.cfg.iterations + 1):
            batch, collect_stats = self.collect()
            stats = self.refit(batch)
            print(
                f"iter {iteration:5d}"
                f"  reward/step {collect_stats['reward_per_step']:7.3f}"
                f"  beta {stats['beta']:8.4f}  kl {stats['kl']:.2f}"
                f"  denoise {stats['denoise_loss']:.4f}  value {stats['value_loss']:.4f}"
                f"  diversity {stats['diversity']:7.2f}  cond_gap {stats['conditioning_gap']:+.4f}"
            )
            if "episode_return" in collect_stats:
                print(f"           episode return {collect_stats['episode_return']:.1f}")
        torch.save({"config": asdict(self.cfg), "model": self.model.state_dict()}, self.cfg.save_path)
        print(f"saved {self.cfg.save_path}")


def parse_config():
    """Every scalar field of Config becomes a --flag. Tuples are set in config.py."""
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for field in fields(Config):
        value = getattr(defaults, field.name)
        if not isinstance(value, tuple):
            parser.add_argument(f"--{field.name}", type=type(value), default=value)
    return Config(**vars(parser.parse_args()))


if __name__ == "__main__":
    Trainer(parse_config()).train()
