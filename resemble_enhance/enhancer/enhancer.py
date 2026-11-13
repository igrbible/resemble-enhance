import logging

import torch
from torch import Tensor, nn
from torch.distributions import Beta

from ..common import Normalizer
from ..denoiser.denoiser import Denoiser
from ..denoiser.hparams import HParams as DenoiserHParams
from ..melspec import MelSpectrogram
from .hparams import HParams
from .lcfm import CFM, IRMAE, LCFM
from .univnet import UnivNet

logger = logging.getLogger(__name__)


def _maybe(fn):
    def _fn(*args):
        if args[0] is None:
            return None
        return fn(*args)

    return _fn


def _normalize_wav(x: Tensor):
    return x / (x.abs().max(dim=-1, keepdim=True).values + 1e-7)


class Enhancer(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp

        n_mels = self.hp.num_mels
        vocoder_input_dim = n_mels + self.hp.vocoder_extra_dim
        latent_dim = self.hp.lcfm_latent_dim

        self.lcfm = LCFM(
            IRMAE(
                input_dim=n_mels,
                output_dim=vocoder_input_dim,
                latent_dim=latent_dim,
            ),
            CFM(
                cond_dim=n_mels,
                output_dim=self.hp.lcfm_latent_dim,
                solver_nfe=self.hp.cfm_solver_nfe,
                solver_method=self.hp.cfm_solver_method,
                time_mapping_divisor=self.hp.cfm_time_mapping_divisor,
            ),
            z_scale=self.hp.lcfm_z_scale,
        )

        self.lcfm.set_mode_(self.hp.lcfm_training_mode)

        self.mel_fn = MelSpectrogram(hp)
        self.vocoder = UnivNet(self.hp, vocoder_input_dim)
        self.denoiser = Denoiser(DenoiserHParams())     # its weights come with the enhancer's own checkpoint
        self.normalizer = Normalizer()

        self._eval_lambd = 0.0

        self.dummy: Tensor
        self.register_buffer("dummy", torch.zeros(1))


    def to_mel(self, x: Tensor, drop_last=True):
        """
        Args:
            x: (b t), wavs
        Returns:
            o: (b c t), mels
        """
        if drop_last:
            return self.mel_fn(x)[..., :-1]  # (b d t)
        return self.mel_fn(x)

    def _may_denoise(self, x: Tensor, y: Tensor | None = None):
        if self.hp.lcfm_training_mode == "cfm":
            return self.denoiser(x, y)
        return x

    def configurate_(self, nfe, solver, lambd, tau):
        """
        Args:
            nfe: number of function evaluations
            solver: solver method
            lambd: denoiser strength [0, 1]
            tau: prior temperature [0, 1]
        """
        self.lcfm.cfm.solver.configurate_(nfe, solver)
        self.lcfm.eval_tau_(tau)
        self._eval_lambd = lambd

    def forward(self, x: Tensor, y: Tensor | None = None, z: Tensor | None = None):
        """
        Args:
            x: (b t), mix wavs (fg + bg)
            y: (b t), fg clean wavs
            z: (b t), fg distorted wavs
        Returns:
            o: (b t), reconstructed wavs
        """
        assert x.dim() == 2, f"Expected (b t), got {x.size()}"
        assert y is None or y.dim() == 2, f"Expected (b t), got {y.size()}"

        if self.hp.lcfm_training_mode == "cfm":
            self.normalizer.eval()

        x = _normalize_wav(x)
        y = _maybe(_normalize_wav)(y)
        z = _maybe(_normalize_wav)(z)

        x_mel_original = self.normalizer(self.to_mel(x), update=False)  # (b d t)

        if self.hp.lcfm_training_mode == "cfm":
            if self.training:
                lambd = Beta(0.2, 0.2).sample(x.shape[:1]).to(x.device)
                lambd = lambd[:, None, None]
                x_mel_denoised = self.normalizer(self.to_mel(self._may_denoise(x, z)), update=False)
                x_mel_denoised = x_mel_denoised.detach()
                x_mel_denoised = lambd * x_mel_denoised + (1 - lambd) * x_mel_original
            else:
                lambd = self._eval_lambd
                if lambd == 0:
                    x_mel_denoised = x_mel_original
                else:
                    x_mel_denoised = self.normalizer(self.to_mel(self._may_denoise(x, z)), update=False)
                    x_mel_denoised = x_mel_denoised.detach()
                    x_mel_denoised = lambd * x_mel_denoised + (1 - lambd) * x_mel_original
        else:
            x_mel_denoised = x_mel_original

        y_mel = _maybe(self.to_mel)(y)  # (b d t)
        y_mel = _maybe(self.normalizer)(y_mel)

        if self.hp.force_gaussian_prior:
            lcfm_decoded = self.lcfm(x_mel_denoised, y_mel, ψ0=None)  # (b d t)
        else:
            lcfm_decoded = self.lcfm(x_mel_denoised, y_mel, ψ0=x_mel_original)  # (b d t)

        if lcfm_decoded is None:
            o = None
        else:
            o = self.vocoder(lcfm_decoded, y)

        return o
