from dataclasses import dataclass


@dataclass
class Config:
    # Policy: an MLP from observation to (action mean, action log-std).
    hidden_sizes: tuple = (32, 32)

    # History and segments. A sampled policy controls the pendulum for
    # L = segment_multiple * K steps, then a new one is sampled.
    K: int = 8
    segment_multiple: int = 4

    # Pendulums. Mass and length are drawn per episode and hidden from the policy.
    mass_range: tuple = (0.5, 2.0)
    length_range: tuple = (0.5, 1.5)
    max_steps: int = 200
    gamma: float = 0.99
    reward_scale: float = 0.05  # applied to rewards in histories, returns and V

    # Improvement step
    num_envs: int = 64  # N, one sampled policy per environment per iteration
    eps_kl: float = 1.0  # KL target for beta; must lie in (0, log N)

    # Diffusion
    diffusion_steps: int = 100  # T used in training
    sampling_steps: int = 20  # DDIM steps used when sampling
    clip_x0: float = 10.0  # clamp on the predicted clean sample while sampling

    # Networks
    embed_dim: int = 64
    encoder_layers: int = 2
    encoder_heads: int = 4
    unet_channels: int = 32  # U-Net base channels; must be a multiple of 8

    # Optimisation
    lr: float = 3e-4
    minibatch: int = 64
    grad_steps: int = 200  # G
    value_coef: float = 0.5
    bootstrap_size: int = 4096  # M0 random initialisations
    bootstrap_steps: int = 2000
    iterations: int = 1000

    # Run
    device: str = "cpu"
    seed: int = 0
    save_path: str = "sampler.pt"

    @property
    def segment_length(self):
        return self.segment_multiple * self.K
