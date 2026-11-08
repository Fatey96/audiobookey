"""Voice manager: named voices stored in data/voices.json.

Schema (validated in code; data/voices.schema.json mirrors it):

    {
      "narrator": {"engine": "draft", "voice_id": "af_heart"},
      "clone1":   {"engine": "final", "clone": "samples/me.wav",
                   "exaggeration": 0.5, "temperature": 0.8, "cfg_weight": 0.5,
                   "seed": 123}
    }

- engine "draft" -> Kokoro; requires "voice_id".
- engine "final" -> Chatterbox; "clone" (path to a 5-15 s reference wav) is
  optional - without it Chatterbox uses its default voice. "seed" pins the
  sampled voice when no clone sample is given.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

VOICE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

ENGINES = ("draft", "final")


class VoiceError(ValueError):
    pass


def find_data_dir() -> Path:
    """data/ next to the repo root (cwd first, then package parents)."""
    cwd = Path.cwd() / "data"
    if cwd.is_dir():
        return cwd
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
    return Path.cwd() / "data"


def voices_path(data_dir: str | Path | None = None) -> Path:
    base = Path(data_dir) if data_dir else find_data_dir()
    return base / "voices.json"


def validate_voice(name: str, cfg: dict) -> None:
    if not VOICE_NAME_RE.match(name):
        raise VoiceError(f"invalid voice name: {name!r}")
    if not isinstance(cfg, dict):
        raise VoiceError(f"voice {name!r} must map to an object")
    engine = cfg.get("engine")
    if engine not in ENGINES:
        raise VoiceError(
            f"voice {name!r}: engine must be one of {', '.join(ENGINES)}"
        )
    if engine == "draft" and not cfg.get("voice_id"):
        raise VoiceError(f"voice {name!r}: draft voices need a 'voice_id'")
    if engine == "final" and cfg.get("voice_id"):
        raise VoiceError(
            f"voice {name!r}: final voices use 'clone' (sample path) or no "
            "voice id, not a kokoro voice_id"
        )
    if "clone" in cfg and not Path(cfg["clone"]).is_file():
        raise VoiceError(
            f"voice {name!r}: clone sample not found: {cfg['clone']}"
        )
    for key in ("exaggeration", "temperature", "cfg_weight"):
        if key in cfg and not isinstance(cfg[key], (int, float)):
            raise VoiceError(f"voice {name!r}: {key} must be a number")
    if "seed" in cfg and not isinstance(cfg["seed"], int):
        raise VoiceError(f"voice {name!r}: seed must be an integer")


def load_voices(path: str | Path | None = None) -> dict[str, dict]:
    path = Path(path) if path else voices_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VoiceError(f"{path}: top level must be an object")
    for name, cfg in data.items():
        validate_voice(name, cfg)
    return data


def save_voices(voices: dict[str, dict], path: str | Path | None = None) -> Path:
    path = Path(path) if path else voices_path()
    for name, cfg in voices.items():
        validate_voice(name, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(voices, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def resolve_voice(
    name: str,
    quality: str,
    path: str | Path | None = None,
    voices: dict[str, dict] | None = None,
) -> dict:
    """Resolve a --voice argument to a full voice config.

    Unknown names on the draft engine are treated as raw Kokoro voice ids
    (P1 behaviour: --voice am_michael etc.). On the final engine an unknown
    name is an error.
    ``voices`` optionally supplies a table already loaded with load_voices()
    so a convert run reads voices.json once instead of per item.
    """
    if voices is None:
        voices = load_voices(path)
    if name in voices:
        return dict(voices[name])
    if quality == "draft":
        return {"engine": "draft", "voice_id": name}
    shown = Path(path) if path else voices_path()
    raise VoiceError(
        f"voice {name!r} not found in {shown}; "
        "add it with 'audiobookey voices add'"
    )


def voice_fingerprint(cfg: dict, cache: dict | None = None) -> str:
    """Stable hash of a voice config; clone samples are hashed by content.

    ``cache`` — the shared per-convert-run dict — memoizes the clone sample's
    content hash per (path, mtime) so repeated items skip the re-hash, while
    a rewritten sample (new mtime or content) misses the cache and hashes to
    a new key.
    """
    parts = {
        "engine": cfg.get("engine"),
        "voice_id": cfg.get("voice_id"),
        "exaggeration": cfg.get("exaggeration"),
        "temperature": cfg.get("temperature"),
        "cfg_weight": cfg.get("cfg_weight"),
        "seed": cfg.get("seed"),
    }
    if cfg.get("clone"):
        sample = Path(cfg["clone"])
        parts_clone = None
        cache_key = None
        if cache is not None:
            cache_key = (str(sample), sample.stat().st_mtime)
            parts_clone = cache.get(cache_key)
        if parts_clone is None:
            digest = hashlib.sha256()
            with open(sample, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            parts_clone = digest.hexdigest()
            if cache is not None:
                cache[cache_key] = parts_clone
        parts["clone_sha256"] = parts_clone
    canonical = json.dumps(parts, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
