"""Kokoro-82M ONNX TTS engine wrapper with CUDA->CPU fallback.

Windows note: onnxruntime-gpu must find cuDNN/cuBLAS/cuDART/cuFFT DLLs before
the CUDA execution provider can load. We register directories from, in order:

1. ``AUDIOBOOKEY_CUDA_DLL_DIR`` (path list, like PATH)
2. pip-installed ``nvidia-*`` CUDA wheel directories in the active environment
3. any importable torch installation's ``torch/lib`` (bundles the CUDA 12 DLLs)
4. standard CUDA toolkit and cuDNN install paths under Program Files

If the CUDA execution provider cannot be loaded or does not activate, we fall
back to CPU with a warning - the pipeline must always work.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import warnings
from pathlib import Path

import numpy as np

MODELS_DIR_NAME = "models"

_CUDA_DLL_NAMES = (
    "cudart64_12.dll", "cudart64_13.dll", "cublas64_12.dll", "cublas64_13.dll",
    "cublasLt64_12.dll", "cudnn64_9.dll", "cufft64_11.dll", "cufft64_12.dll",
)


class EngineNotAvailable(RuntimeError):
    pass


def _dll_dir_has_cuda_dlls(path: Path) -> bool:
    try:
        entries = {e.name.lower() for e in path.iterdir()}
    except OSError:
        return False
    return any(name in entries for name in _CUDA_DLL_NAMES)


def _candidate_dll_dirs() -> list[Path]:
    dirs: list[Path] = []

    env = os.environ.get("AUDIOBOOKEY_CUDA_DLL_DIR")
    if env:
        dirs.extend(Path(p) for p in env.split(os.pathsep) if p.strip())

    # pip-installed nvidia wheels ship their binaries in package dirs
    for dist in list(importlib.metadata.distributions()):
        name = (dist.metadata.get("Name") or "").lower()
        if not name.startswith("nvidia-"):
            continue
        files = dist.files or []
        seen: set[Path] = set()
        for f in files:
            if str(f).lower().endswith((".dll", ".so")):
                p = Path(dist.locate_file(f)).parent
                if p not in seen:
                    seen.add(p)
                    dirs.append(p)

    # torch bundles a full CUDA 12 runtime in torch/lib
    try:
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec and torch_spec.submodule_search_locations:
            torch_root = Path(next(iter(torch_spec.submodule_search_locations)))
            dirs.append(torch_root / "lib")
    except (ImportError, ValueError):
        pass

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        dirs.extend(sorted(cuda_root.glob("v*/bin")))
    cudnn_root = Path(program_files) / "NVIDIA" / "CUDNN"
    if cudnn_root.is_dir():
        dirs.extend(sorted(cudnn_root.glob("v*/bin")))

    return dirs


def register_cuda_dll_dirs() -> list[Path]:
    """Add directories containing CUDA DLLs to the process search path.

    Static imports of onnxruntime_providers_cuda.dll resolve through PATH on
    Windows (os.add_dll_directory is not consulted for static dependencies),
    so the directories are prepended to PATH as well.
    """
    added: list[Path] = []
    seen: set[str] = set()
    for d in _candidate_dll_dirs():
        if d.is_dir() and _dll_dir_has_cuda_dlls(d):
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass
            added.append(d)
    if added:
        prepend = os.pathsep.join(str(d) for d in reversed(added))
        os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")
    return added


def build_voicepack(voices_dir: Path) -> Path:
    """Combine voice style files into one loadable npz for kokoro-onnx.

    Handles both formats: the official multi-voice bundle (a named npz whose
    entries are (n, 1, 256) style arrays) and raw per-voice float32 dumps from
    HuggingFace (a flat (n, 256) matrix with no header), which are reshaped to
    the (n, 1, 256) layout kokoro-onnx expects. The combined file is rebuilt
    when any source file is newer than it.
    """
    pack = voices_dir / "_voicepack.npz"
    bins = sorted(voices_dir.glob("*.bin"))
    styles: dict[str, np.ndarray] = {}
    for b in bins:
        try:
            loaded = np.load(str(b))
        except Exception:
            loaded = None
        if loaded is not None and hasattr(loaded, "files"):  # npz bundle
            for name in loaded.files:
                styles[name] = np.asarray(loaded[name], dtype=np.float32)
            continue
        raw = np.fromfile(str(b), dtype=np.float32)
        if raw.size and raw.size % 256 == 0:
            styles[b.stem] = raw.reshape(-1, 1, 256).astype(np.float32)
    if not styles:
        raise EngineNotAvailable(
            f"no usable voice files (.bin) found in {voices_dir}"
        )

    if pack.is_file():
        if pack.stat().st_mtime > max(b.stat().st_mtime for b in bins):
            return pack
    np.savez(str(pack), **styles)
    return pack


_CANARY = "Hello, this is a quick test of the audio engine."


def _canary(kokoro, voice: str, speed: float = 1.0, lang: str = "en-us") -> None:
    """Synthesize a short sentence in memory; fail on empty or silent output.

    Some onnxruntime/model combinations (notably fp16 + CUDA EP) load fine and
    simply return zero samples - only a real synthesis reveals the problem.
    """
    samples, sample_rate = kokoro.create(_CANARY, voice=voice, speed=speed, lang=lang)
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size < sample_rate // 4 or np.abs(samples).max() < 1e-4:
        raise RuntimeError(
            f"canary synthesis produced {samples.size} silent samples"
        )


class KokoroTTSEngine:
    """Wraps kokoro-onnx with an explicit execution-provider choice.

    Candidate configurations are tried in order until one passes a canary
    synthesis (fast, in-memory, no files written):

        model_fp16 + CUDA  ->  model_q4 + CUDA  ->  model_fp16 + CPU

    This matters because the fp16 graph can load on the CUDA execution
    provider yet silently produce zero-length audio on some onnxruntime
    builds; only a real synthesis proves a configuration works.
    """

    # (model_file, prefer_gpu) candidates, best first
    _CANDIDATES = [("model_fp16.onnx", True), ("model_q4.onnx", True),
                   ("model_fp16.onnx", False)]

    def __init__(
        self,
        models_dir: str | Path,
        model_file: str | None = None,
        prefer_gpu: bool = True,
        voice: str = "af_heart",
    ) -> None:
        self._default_voice = voice
        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        models_dir = Path(models_dir)
        voices_dir = models_dir / "voices"
        if not voices_dir.is_dir():
            raise EngineNotAvailable(f"voices directory not found: {voices_dir}")

        register_cuda_dll_dirs()
        cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()

        pack = build_voicepack(voices_dir)
        try:
            kokoro_version = importlib.metadata.version("kokoro-onnx")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover
            kokoro_version = "unknown"
        self._kokoro_version = kokoro_version

        if model_file is not None:
            candidates = [(model_file, prefer_gpu)]
        else:
            candidates = []
            if prefer_gpu and cuda_available:
                candidates = [(m, True) for m, g in self._CANDIDATES if g]
                candidates += [(m, False) for m, g in self._CANDIDATES if not g]
            else:
                candidates = [(m, g) for m, g in self._CANDIDATES if not g][:1]

        attempt_errors: list[str] = []
        for candidate_model, candidate_gpu in candidates:
            device = "?"
            model_path = models_dir / candidate_model
            if not model_path.is_file():
                attempt_errors.append(f"{candidate_model}: file missing")
                continue
            try:
                if candidate_gpu and cuda_available:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    if "CUDAExecutionProvider" not in session.get_providers():
                        device = "cpu"
                        warnings.warn(
                            "CUDA execution provider did not activate; "
                            "running on CPU.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    else:
                        device = "cuda"
                else:
                    session = ort.InferenceSession(
                        str(model_path), providers=["CPUExecutionProvider"]
                    )
                    device = "cpu"

                kokoro = (
                    Kokoro.from_session(session, str(pack))
                    if session is not None
                    else Kokoro(str(model_path), str(pack))
                )
                _canary(kokoro, voice)  # raises on silent/empty output
                if prefer_gpu and not candidate_gpu:
                    warnings.warn(
                        "Falling back to CPU execution provider.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                self.kokoro = kokoro
                self.device = device
                self._model_file = candidate_model
                if len(candidates) > 1:
                    warnings.warn(
                        f"Using {candidate_model} with {device.upper()} "
                        "execution provider after validating a test "
                        "synthesis.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                if voice not in self.kokoro.get_voices():
                    raise EngineNotAvailable(
                        f"voice {voice!r} not in voicepack (available: "
                        f"{', '.join(self.kokoro.get_voices())})"
                    )
                return
            except Exception as exc:
                attempt_errors.append(
                    f"{candidate_model}+{device}: {type(exc).__name__}: "
                    f"{str(exc)[:160]}"
                )

        raise EngineNotAvailable(
            "no working Kokoro ONNX configuration found; attempts:\n  "
            + "\n  ".join(attempt_errors)
        )

    @property
    def engine_version(self) -> str:
        """Identifies the synthesizer: anything it changes invalidates cache."""
        return (
            f"kokoro-onnx=={self._kokoro_version};model={self._model_file};"
            f"ep={self.device}"
        )

    def synthesize(
        self,
        text: str,
        voice_config: dict | None = None,
        speed: float = 1.0,
        lang: str | None = None,
        gap: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """Synthesize one chunk. Returns (float32 mono samples, sample_rate).

        voice_config carries ``voice_id`` (Kokoro voice name). ``gap`` is the
        sentence pause in seconds; None keeps the kokoro default.
        """
        cfg = voice_config or {}
        voice_id = cfg.get("voice_id", self._default_voice)
        kwargs = {}
        if gap is not None:
            kwargs["sentence_pause"] = gap
        samples, sample_rate = self.kokoro.create(
            text, voice=voice_id, speed=speed, lang=lang, **kwargs
        )
        return np.asarray(samples, dtype=np.float32), int(sample_rate)


def find_models_dir(explicit: str | Path | None = None) -> Path:
    """Locate the models/ directory: explicit > env > cwd > package parents."""
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("AUDIOBOOKEY_MODELS_DIR")
    if env:
        return Path(env)
    cwd = Path.cwd() / MODELS_DIR_NAME
    if (cwd / "voices").is_dir():
        return cwd
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / MODELS_DIR_NAME
        if (candidate / "voices").is_dir():
            return candidate
    raise EngineNotAvailable(
        "Could not locate the models/ directory. Set AUDIOBOOKEY_MODELS_DIR "
        "or run from the repository root."
    )
