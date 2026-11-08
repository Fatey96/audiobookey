"""Real TTS smoke test using the smallest Kokoro model (model_q4.onnx).

Skipped automatically when models/model_q4.onnx is not present, so the unit
suite stays runnable anywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from audiobookey.engine import KokoroTTSEngine, find_models_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _q4_model_available() -> bool:
    try:
        models_dir = find_models_dir(REPO_ROOT / "models")
    except Exception:
        return False
    return (models_dir / "model_q4.onnx").is_file()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.skipif(not _q4_model_available(), reason="model_q4.onnx not downloaded")
def test_kokoro_engine_synthesizes(tmp_path):
    engine = KokoroTTSEngine(
        find_models_dir(REPO_ROOT / "models"), model_file="model_q4.onnx"
    )
    samples, sample_rate = engine.synthesize(
        "This is a smoke test sentence.",
        voice_config={"voice_id": "af_heart"},
        speed=1.0,
        lang="en-us",
    )
    assert sample_rate > 0
    assert samples.dtype == __import__("numpy").float32
    assert samples.size > sample_rate // 2  # at least half a second of audio
