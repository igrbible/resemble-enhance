# Ruach Studio's derivative of Resemble Enhance

This branch, `ruach-studio`, is the code [Ruach Studio](https://github.com/igrbible/Ruach_Studio) derives from [resemble-ai/resemble-enhance](https://github.com/resemble-ai/resemble-enhance). It starts at upstream commit `8e978149bfe8` (03.12.2024), the exact commit the Studio vendored, and adds our changes on top as plain commits. We hide nothing and are not ashamed of it: what the Studio runs is here for anyone to read, compare and take back.

The Studio carries only the inference part of Resemble Enhance (its "TTS Voice Upscaler" step for the Narrator's spoken takes). This branch keeps the whole upstream tree, training, data and gradio parts included, and changes only what the Studio's copy changes:

- `resemble_enhance/enhancer/lcfm/cfm.py`: `float(scipy.optimize.fsolve(...)[0])`, since `float()` of a one-element array raises under NumPy 2; the matplotlib and tqdm imports dropped.
- `resemble_enhance/enhancer/lcfm/irmae.py`: `IRMAE.encode` no longer computes the debug quantiles of `z` (a device sync per chunk, and `quantile()` refuses bf16/fp16).
- `resemble_enhance/enhancer/hparams.py`: the released `enhancer_stage2` run's values written in as defaults, so nothing is read from disk.
- `resemble_enhance/enhancer/enhancer.py`: the denoiser built in place (its weights come in the enhancer's own checkpoint) instead of loaded from a run dir; stage-1 pretrained loading, the pandas summary and the matplotlib visualisation removed.
- `resemble_enhance/inference.py`: `range` instead of tqdm's `trange`.
- `resemble_enhance/ruach/api.py` (new): `load`, `denoise`, `enhance` and `convert` as the Studio calls them, float32 always, the weights read from a safetensors file converted from the released checkpoint. No weights are in this branch.

The Studio's own copy also drops the files inference never reads (training, data, distorters, gradio) and the omegaconf/rich parts of the root `hparams.py`; those removals are not repeated here. Because of the `enhancer.py` change this branch is meant for inference: to train, use upstream `main`.

HERESY numbers (the Studio's numbered patch log): **2312** (the TTS Voice Upscaler, Resemble Enhance vendored, in Ruach Studio's lab commit `bd58a7a`) and **2311** (the same commit's Reader work, not in this tree).

## Licences

- Resemble Enhance: MIT, Copyright (c) 2023 Resemble AI (`LICENSE`). The alias-free-torch parts under `resemble_enhance/enhancer/univnet/alias_free_torch/` are from [junjun3518/alias-free-torch](https://github.com/junjun3518/alias-free-torch), Apache License 2.0.
- Our changes: as Ruach Studio's release states for what we wrote, the GNU Affero General Public License, version 3 or any later version, with the additional terms of its section 7 (see the Studio's `LICENSE` and `LICENSE-ADDITIONAL-TERMS.md`). The upstream files keep their MIT notice.

Ruach Studio is not made, endorsed or supported by Resemble AI; "Resemble" is their name, and the Studio's step is called TTS Voice Upscaler.
