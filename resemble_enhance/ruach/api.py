"""HERESY 2312 (Viktor 09.11.2026: «TTS Voice Upscaler… Resemble Enhance под капот… строго в мире Чтеца»; «Resemble Enhance approved with all
node defaults, in f32 only»): the Studio's way into Resemble Enhance. One model per process (the step is a process of its own, one job), float32
always; its weights from enhancer_stage2.safetensors (fp16 as released, made by convert() from the deepspeed checkpoint) or that checkpoint.

    load(model_dir, device) -> Enhancer       denoise(model, wav, sr, device, chunk_s) -> (wav, 44100)
    enhance(model, wav, sr, device, nfe, solver, lambd, tau, chunk_s) -> (wav, 44100)
    convert(model_dir) -> path of the safetensors written (sha256 beside it)"""
import hashlib
from pathlib import Path

import torch

from ..enhancer.enhancer import Enhancer
from ..enhancer.hparams import HParams
from ..inference import inference

ST = "enhancer_stage2.safetensors"
CKPT = Path("enhancer_stage2") / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"


def state_of(model_dir):
    d = Path(model_dir)
    if (d / ST).is_file():
        from safetensors.torch import load_file
        return load_file(str(d / ST))
    if (d / CKPT).is_file():
        return torch.load(str(d / CKPT), map_location="cpu", weights_only=True)["module"]
    raise FileNotFoundError(f"no Resemble Enhance weights in {d} ({ST} or {CKPT})")


def load(model_dir, device):
    m = Enhancer(HParams())
    m.load_state_dict({k: v.float() for k, v in state_of(model_dir).items()})
    m.eval().float().to(device)
    return m


@torch.inference_mode()
def denoise(model, wav, sr, device, chunk_s=30.0):
    return inference(model=model.denoiser, dwav=wav, sr=sr, device=device, chunk_seconds=chunk_s)


@torch.inference_mode()
def enhance(model, wav, sr, device, nfe=64, solver="midpoint", lambd=0.1, tau=0.5, chunk_s=30.0):
    if not (0 < nfe <= 128 and solver in ("midpoint", "rk4", "euler") and 0 <= lambd <= 1 and 0 <= tau <= 1):
        raise ValueError(f"nfe {nfe}, solver {solver}, lambd {lambd}, tau {tau}: out of range")
    model.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
    return inference(model=model, dwav=wav, sr=sr, device=device, chunk_seconds=chunk_s)


def convert(model_dir):
    """The released deepspeed checkpoint as safetensors (its own fp16 tensors, nothing cast), and its sha256 in a file beside it."""
    from safetensors.torch import save_file
    d = Path(model_dir)
    sd = torch.load(str(d / CKPT), map_location="cpu", weights_only=True)["module"]
    out = d / ST
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out), metadata={"source": "ResembleAI/resemble-enhance enhancer_stage2", "license": "MIT"})
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    (d / (ST + ".sha256")).write_text(f"{h}  {ST}\n")
    return out
