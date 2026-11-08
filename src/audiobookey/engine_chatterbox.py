"""Chatterbox engine -- the `--quality final` synthesizer.

Notes
-----
- Model files are downloaded (resumably, via huggingface_hub) into
  models/chatterbox/ on first use.
- Chatterbox is a CUDA/PyTorch model; if CUDA is unavailable it degrades to
  CPU with a warning (slow but functional).
- Chatterbox synthesizes at its own S3GEN_SR sample rate (currently 24 kHz);
  the pipeline resamples page WAVs uniformly before concatenation, so draft
  and final outputs stay interchangeable.
- Chatterbox has no speed control; the CLI rejects non-1.0 speed values with
  --quality final rather than silently ignoring them.
"""

from __future__ import annotations

import importlib.metadata
import warnings
from pathlib import Path

import numpy as np

from .engine import EngineNotAvailable, register_cuda_dll_dirs

REPO_ID = "ResembleAI/chatterbox"
_MODEL_FILES = (
    "ve.safetensors",
    "t3_cfg.safetensors",
    "s3gen.safetensors",
    "tokenizer.json",
    "conds.pt",  # built-in default voice
)

TARGET_SAMPLE_RATE = 24000


def download_model(models_dir: str | Path, reporter=None) -> Path:
    """Ensure the chatterbox checkpoints exist in models/chatterbox (resumable)."""
    from huggingface_hub import hf_hub_download

    ckpt_dir = Path(models_dir) / "chatterbox"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name in _MODEL_FILES:
        target = ckpt_dir / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        if reporter is not None:
            reporter.phase(f"Downloading chatterbox model file: {name}...")
        hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(ckpt_dir))
    return ckpt_dir


class ChatterboxEngine:
    """`--quality final` synthesizer. synthesize() returns (float32 samples, sr)."""

    def __init__(
        self,
        models_dir: str | Path,
        prefer_gpu: bool = True,
        device: str | None = None,
    ) -> None:
        import torch
        from chatterbox import ChatterboxTTS

        register_cuda_dll_dirs()
        ckpt_dir = download_model(models_dir)
        for name in _MODEL_FILES:
            if not (ckpt_dir / name).is_file() or (ckpt_dir / name).stat().st_size == 0:
                raise EngineNotAvailable(f"chatterbox checkpoint missing: {name}")

        if device is None:
            device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
            if prefer_gpu and device == "cpu":
                warnings.warn(
                    "CUDA is not available for torch; running Chatterbox "
                    "on CPU (slow).",
                    RuntimeWarning,
                    stacklevel=2,
                )
        try:
            self.tts = ChatterboxTTS.from_local(str(ckpt_dir), device)
        except Exception as exc:
            if device == "cuda":
                warnings.warn(
                    f"Chatterbox failed on CUDA ({exc}); falling back to CPU.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                device = "cpu"
                self.tts = ChatterboxTTS.from_local(str(ckpt_dir), device)
            else:
                raise

        self.device = device
        self.sample_rate = int(self.tts.sr)
        self.engine_name = "chatterbox"
        self.quality = "final"
        self.model_variant = "t3_cfg-v1.0"
        try:
            self._version = importlib.metadata.version("chatterbox-tts")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            self._version = "unknown"

        # Canary: prove the device combination actually produces audio.
        samples, _ = self.synthesize("A quick test of the audio engine.")
        if samples.size == 0:
            raise EngineNotAvailable("chatterbox canary produced no audio")

    @property
    def engine_version(self) -> str:
        return (
            f"chatterbox-tts=={self._version};model={self.model_variant};"
            f"dev={self.device};quality=final"
        )

    def synthesize(
        self,
        text: str,
        voice_config: dict | None = None,
        speed: float = 1.0,
        lang: str | None = None,
        gap: float | None = None,
    ) -> tuple[np.ndarray, int]:
        import torch

        cfg = dict(voice_config or {})
        seed = cfg.get("seed")
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        clone = cfg.get("clone")
        wav = self.tts.generate(
            text,
            audio_prompt_path=str(clone) if clone else None,
            exaggeration=float(cfg.get("exaggeration", 0.5)),
            cfg_weight=float(cfg.get("cfg_weight", 0.5)),
            temperature=float(cfg.get("temperature", 0.8)),
        )
        samples = np.asarray(wav.detach().cpu().float().numpy(), dtype=np.float32)
        return samples.reshape(-1), int(self.sample_rate)
