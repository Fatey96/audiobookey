"""End-to-end PDF -> m4b/mp3 conversion pipeline.

P2 additions over P1: lexicon substitution, pause + [voice:name] markup,
voice maps (regex -> voice), named voices from data/voices.json, and
Chatterbox (`--quality final`) as a second engine coexisting with Kokoro.

C5A: per-page assembly overlaps the next page's synthesis (single
assembler thread), and the page wavs feed the loudnorm pass through a
concat list -- the intermediate book.wav is gone.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import markup
from . import voices as voices_mod

from . import audio as audio_pipeline
from .audio import (
    write_wav,
    write_silence,
    concat_wavs,
    read_wav_duration,
    read_wav_sample_rate,
    resample_wav,
    encode_m4b,
    encode_mp3,
    copy_wav,
)
from .cache import ChunkCache, book_id, chunk_key
from .chunk import PhonemeCounter, chunk_text
from .clean import clean_pages
from .engine import find_models_dir
from .engine_chatterbox import TARGET_SAMPLE_RATE
from .engine_pool import make_engine
from .extract import extract_pages
from .lexicon import Lexicon
from .markup import (
    PauseSegment,
    Segment,
    SpeechSegment,
    VoiceMap,
    split_segments,
)
from .markup import (
    PauseSegment,
    Segment,
    SpeechSegment,
    VoiceMap,
    quote_units,
    split_segments,
)
from .reporter import QuietReporter, Reporter
from .sentences import TimelineChunk, build_sentence_entries, write_manifest
from .voices import load_voices, resolve_voice, voice_fingerprint, voices_path

logger = logging.getLogger("audiobookey")


@dataclass
class ConvertResult:
    m4b: Path
    mp3: Path
    chapters: int
    duration_seconds: float
    skipped_pages: list[int]
    cache_hits: int
    cache_misses: int
    total_pages: int
    total_chunks: int
    engine_version: str = ""
    details: dict = field(default_factory=dict)

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


@dataclass
class _Item:
    """One renderable unit of a page: a speech stretch or a silence."""

    kind: str  # "speech" | "pause"
    text: str = ""
    seconds: float = 0.0
    voice_name: str | None = None
    voice_cfg: dict | None = None
    chunks: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    wavs: list[Path] = field(default_factory=list)


@dataclass
class _PagePlan:
    number: int
    items: list[_Item]


def _assemble_page(plan: _PagePlan, work_dir: Path, target_sr: int) -> Path:
    """Build work_dir / page-NNNN.wav from the plan's item WAVs.

    Silence pauses are written and speech chunks are resampled to the
    uniform target sample rate, then concatenated in reading order.
    """
    parts: list[Path] = []
    for item in plan.items:
        if item.kind == "pause":
            if item.seconds > 0.005:
                p = work_dir / f"pause-{plan.number}-{len(parts)}.wav"
                write_silence(p, item.seconds, target_sr)
                parts.append(p)
            continue
        wavs = [
            _uniform_rate(w, work_dir, target_sr)
            for w in item.wavs
        ]
        if len(wavs) == 1:
            parts.append(wavs[0])
        else:
            p = work_dir / f"group-{plan.number}-{len(parts)}.wav"
            parts.append(concat_wavs(wavs, p))
    dest = work_dir / f"page-{plan.number:04d}.wav"
    if len(parts) == 1:
        copy_wav(parts[0], dest)
    else:
        concat_wavs(parts, dest)
    return dest


def _split_paragraph_segments(text: str, voice: str | None) -> list[Segment]:
    """Blank lines become paragraph-pause separated speech items."""
    out: list[Segment] = []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    for i, para in enumerate(paragraphs):
        if i:
            out.append(PauseSegment(seconds=markup.PARAGRAPH_PAUSE))
        out.append(SpeechSegment(text=para.strip(), voice=voice))
    return out


def _page_plan_items(
    page_text: str,
    diarize: bool = False,
    dialogue_voice: str | None = None,
) -> list[Segment]:
    """Markup parse + paragraph expansion for one cleaned page.

    With diarize enabled, speech is further split at quote boundaries so
    quoted spans can carry the dialogue voice; explicit [voice:] markup
    always wins over diarize.
    """
    items: list[Segment] = []
    for seg in split_segments(page_text):
        if isinstance(seg, PauseSegment):
            items.append(seg)
            continue
        pieces = (
            [(seg.text, False)]
            if seg.voice is not None or not diarize
            else quote_units(seg.text)
        )
        for piece, is_quoted in pieces:
            if not piece.strip():
                continue
            voice = dialogue_voice if is_quoted else seg.voice
            items.extend(_split_paragraph_segments(piece.strip(), voice))
    return items


def _engine_device_line(kind: str, eng: object) -> str | None:
    """C4 W3: one explicit device line for the convert run log.

    Engines expose their device as a plain attribute (kokoro: execution
    provider, chatterbox: torch device); a silent CPU fallback is otherwise
    only a RuntimeWarning. Engines without a device (test doubles) emit
    nothing.
    """
    device = getattr(eng, "device", None)
    if not device:
        return None
    name = "chatterbox" if kind == "final" else "kokoro"
    return f"engine: {name} on {device}"


def convert(
    pdf_path: str | Path,
    out_dir: str | Path,
    voice: str = "af_heart",
    speed: float = 1.0,
    quality: str = "draft",
    encode_tier: str = "standard",
    lang: str = "en-us",
    models_dir: str | Path | None = None,
    engine: object | None = None,
    reporter: Reporter | None = None,
    lexicon: Lexicon | None = None,
    voice_map: VoiceMap | None = None,
    gap: float = markup.DEFAULT_SENTENCE_GAP,
    diarize: bool = False,
    cache_root: str | Path | None = None,
) -> ConvertResult:
    """Convert a PDF into an m4b (chapter per page) and an mp3."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    reporter = reporter or QuietReporter()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # -- voices: one read per run (C4 W2) ------------------------------------
    # voices.json is loaded once instead of per item, and the clone-sample
    # fingerprint cache is per run, keyed (path, mtime) so a rewritten sample
    # still invalidates.
    vpath = voices_path()
    voices_table = load_voices(vpath)
    fp_cache: dict[tuple[str, float], str] = {}

    # -- diarize: pick the dialogue voice up front; fail loudly if absent ---
    dialogue_voice_name: str | None = None
    if diarize:
        dialogue_voice_name = next(
            (name for name in voices_table if name.startswith("dialogue")), None
        )
        if dialogue_voice_name is None:
            raise ValueError(
                "--diarize: no dialogue voice found in "
                f"{vpath}; add one whose name starts with 'dialogue'"
            )

    # -- extract ----------------------------------------------------------
    reporter.phase("Extracting text from PDF...")
    raw_pages = extract_pages(pdf_path)
    logger.info("extracted %d pages from %s", len(raw_pages), pdf_path.name)

    # -- clean -------------------------------------------------------------
    cleaned = clean_pages([p.text for p in raw_pages])
    skipped = [p.number for p in cleaned if p.skipped]
    for number in skipped:
        reporter.warning(
            f"page {number}: no extractable text (scanned page?); skipping"
        )
    text_pages = [p for p in cleaned if not p.skipped]
    if not text_pages:
        raise RuntimeError(
            "no extractable text in this PDF; it is likely a scanned book"
        )

    # -- lexicon (before chunking; changes text and thus cache keys) ---------
    if lexicon is not None:
        reporter.phase(f"Applying lexicon ({len(lexicon.rules)} rules)...")
        for page in text_pages:
            page.text = lexicon.apply(page.text)

    # -- markup + chunking --------------------------------------------------
    reporter.phase("Chunking text (pauses + voice markup respected)...")
    counter = PhonemeCounter()
    logger.info("phoneme counter: %s", "espeak" if counter.exact else "estimate")

    plans: list[_PagePlan] = []
    for page in text_pages:
        items = _page_plan_items(page.text, diarize, dialogue_voice_name)
        resolved: list[_Item] = []
        for item in items:
            if isinstance(item, PauseSegment):
                resolved.append(
                    _Item(kind="pause", seconds=item.seconds)
                )
                continue
            voice_name = (
                item.voice
                or (voice_map.match(item.text) if voice_map else None)
                or voice
            )
            cfg = resolve_voice(voice_name, quality, vpath, voices=voices_table)
            # Chatterbox generates with a bounded AR context; keep its chunks
            # clearly inside the known-good range.
            cap = 450 if cfg["engine"] == "draft" else 300
            resolved.append(
                _Item(
                    kind="speech",
                    text=item.text,
                    voice_name=voice_name,
                    voice_cfg=cfg,
                    chunks=chunk_text(item.text, max_phonemes=cap, count=counter),
                )
            )
        plans.append(_PagePlan(number=page.number, items=resolved))

    total_chunks = sum(
        len(item.chunks) for plan in plans for item in plan.items
        if item.kind == "speech"
    )

    # -- synthesis with cache -------------------------------------------------
    bid = book_id(pdf_path)
    # P9 --cache-root: park the per-book cache dir under a shared root when
    # asked, so different --out dirs (and the server's per-job dirs) reuse
    # one durable cache; default keeps chunks under <out>/cache exactly as
    # before. Work/per-page dirs stay under out_dir either way.
    cache_base = out_dir / "cache" if cache_root is None else Path(cache_root)
    cache_dir = cache_base / bid
    chunk_dir = cache_dir / "chunks"
    work_dir = out_dir / "work" / bid
    work_dir.mkdir(parents=True, exist_ok=True)
    chunk_cache = ChunkCache(chunk_dir)

    if any(
        item.voice_cfg and item.voice_cfg["engine"] == "final"
        for plan in plans
        for item in plan.items
    ) and speed != 1.0:
        raise ValueError(
            "speed != 1.0 is not supported by the chatterbox engine "
            "(--quality final); use speed 1.0 or --quality draft"
        )

    models_root = find_models_dir(models_dir)
    loaded_engines: dict[str, object] = {}

    def engine_for(kind: str) -> object:
        if kind not in loaded_engines:
            loaded_engines[kind] = (
                engine if engine is not None else make_engine(kind, models_root)
            )
            eng = loaded_engines[kind]
            line = _engine_device_line(kind, eng)
            if line is not None:
                reporter.phase(line)  # lands in the convert job's run log
        return loaded_engines[kind]

    reporter.start_chunks(total_chunks, len(text_pages))
    timeline_chunks: list[TimelineChunk] = []
    page_wavs: list[Path] = []
    page_numbers: list[int] = []

    # C5A: page assembly (ffmpeg per-page concat) overlaps the synthesis of
    # the following page. One background thread owns ALL assembly work;
    # synthesis stays on this thread because engines are not thread-safe.
    # Results are queued in page order, so page_ready still fires exactly
    # once per page, in page order, followed immediately by that page's
    # single `update` (cache-stats snapshot taken at the end of the page's
    # synthesis, keeping update payloads deterministic).
    assembly_jobs: queue.Queue = queue.Queue()
    assembly_results: queue.Queue = queue.Queue()
    announced = 0

    def _assemble_worker() -> None:
        while True:
            plain_job = assembly_jobs.get()
            if plain_job is None:
                return
            plan, page_timeline, stats = plain_job
            try:
                wav = _assemble_page(plan, work_dir, TARGET_SAMPLE_RATE)
                duration = read_wav_duration(wav)
                error = None
            except Exception as exc:  # surfaced on the main thread
                wav, duration, error = None, 0.0, exc
            assembly_results.put(
                (plan.number, wav, duration, page_timeline, stats, error)
            )

    assembler = threading.Thread(
        target=_assemble_worker, name="page-assembler", daemon=True
    )
    assembler.start()

    def _pump_ready(block: bool = False) -> None:
        """Announce finished pages in order (page_ready, then its update)."""
        nonlocal announced
        while announced < len(plans):
            try:
                finished = assembly_results.get_nowait()
            except queue.Empty:
                if not block:
                    return
                finished = assembly_results.get()
            announced += 1
            number, wav, duration, page_timeline, stats, error = finished
            if error is not None:
                raise error
            sentences = [
                {"start_s": s["start_s"], "end_s": s["end_s"], "text": s["text"]}
                for s in build_sentence_entries(page_timeline)
            ]
            reporter.page_ready(
                number, indexes[number], len(plans), wav, duration, sentences
            )
            # update stays last per page and is cache stats only
            reporter.update(stats[0], stats[1], total_chunks)
            page_wavs.append(wav)
            page_numbers.append(number)
            timeline_chunks.extend(page_timeline)

    indexes: dict[int, int] = {}
    try:
        for index, plan in enumerate(plans, start=1):
            indexes[plan.number] = index
            page_timeline: list[TimelineChunk] = []
            for item in plan.items:
                if item.kind != "speech":
                    if item.seconds > 0.005:
                        page_timeline.append(
                            TimelineChunk(
                                page=plan.number,
                                text="",
                                voice=voice,
                                duration_s=item.seconds,
                            )
                        )
                    continue
                kind = item.voice_cfg["engine"]
                eng = engine_for(kind)
                engine_id = eng.engine_version
                if "quality=" not in engine_id:
                    engine_id += f";quality={kind}"
                fp = voice_fingerprint(item.voice_cfg, cache=fp_cache)
                # gap only affects kokoro synthesis; keep it out of other
                # engines' cache keys
                extra = str(gap) if kind == "draft" else ""
                item.keys = [
                    chunk_key(c, fp, speed, engine_id, extra=extra)
                    for c in item.chunks
                ]
                for key, chunk_str in zip(item.keys, item.chunks):
                    cached = chunk_cache.get(key)
                    if cached is None:
                        samples, sr = _synthesize_with_retry(
                            eng, chunk_str, item.voice_cfg, speed, lang, gap, key
                        )
                        tmp = chunk_dir / f".{key}.new.wav"
                        write_wav(tmp, samples, sr)
                        cached = chunk_cache.store(key, tmp)
                    item.wavs.append(cached)
                    reporter.chunk_done()
                    # a page assembled while earlier chunks were being
                    # synthesized is announced here, between chunks
                    _pump_ready()
                    page_timeline.append(
                        TimelineChunk(
                            page=plan.number,
                            text=chunk_str,
                            voice=item.voice_name or voice,
                            duration_s=read_wav_duration(cached),
                        )
                    )
            # hand this page to the assembler; its WAV is built while the
            # next page's chunks synthesize (also on a cache hit)
            assembly_jobs.put(
                (
                    plan,
                    page_timeline,
                    (chunk_cache.hits, chunk_cache.hits + chunk_cache.misses),
                )
            )
            _pump_ready()
    finally:
        # let an aborting run drain its in-flight assembly, never leak the
        # thread (assembly failure is re-raised via _pump_ready above)
        assembly_jobs.put(None)
        assembler.join()
    # announce pages that finished while the last page was synthesizing
    _pump_ready(block=True)

    # -- chapters + encode ----------------------------------------------------
    reporter.phase("Encoding outputs...")
    durations = [read_wav_duration(p) for p in page_wavs]
    chapters: list[tuple[float, float, str]] = []
    cursor = 0.0
    for number, dur in zip(page_numbers, durations):
        chapters.append((cursor, cursor + dur, f"Page {number}"))
        cursor += dur

    stem = pdf_path.stem
    # P6B encode tiers: medium/high synthesize identical chunk WAVs (same
    # engine/config), so chunk cache keys don't include the tier; only the
    # final encode differs, and each job's outputs live in a fresh per-job
    # directory, so a standard-tier medium output is never reused for high.
    # C4 W1: loudnorm runs once into an intermediate in the per-job work
    # dir; both encodes ship from that single normalized pass. C5A: the
    # pages feed that pass through the concat demuxer -- book.wav is never
    # materialized (single-page runs feed the page WAV directly).
    m4b_path = out_dir / f"{stem}.m4b"
    mp3_path = out_dir / f"{stem}.mp3"
    m4b, mp3 = audio_pipeline.encode_m4b_and_mp3(
        page_wavs, chapters, m4b_path, mp3_path,
        audio_bitrate="128k" if encode_tier == "best" else "64k",
        mp3_bitrate="256k" if encode_tier == "best" else "192k",
        work_dir=work_dir,
    )

    voices_used = sorted(
        {
            item.voice_name
            for plan in plans
            for item in plan.items
            if item.kind == "speech"
        }
    )
    manifest = {
        "book_id": bid,
        "pdf": str(pdf_path),
        "voice": voice,
        "voices_used": voices_used,
        "voice_map": bool(voice_map),
        "speed": speed,
        "quality": quality,
        "lang": lang,
        "gap": gap,
        "engines": sorted(loaded_engines.keys()),
        "total_pages": len(raw_pages),
        "usable_pages": page_numbers,
        "skipped_pages": skipped,
        "chapters": [
            {"title": t, "start": s, "end": e} for s, e, t in chapters
        ],
        "cache": {"hits": chunk_cache.hits, "misses": chunk_cache.misses},
    }
    (cache_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # -- sentence manifest for the reader UI (P4-B) --------------------------
    sen_manifest = write_manifest(
        out_dir / "manifest.json",
        engine="chatterbox" if quality == "final" else "kokoro",
        speed=speed,
        sentences=build_sentence_entries(timeline_chunks),
    )
    logger.info("sentence manifest written: %s", sen_manifest)

    return ConvertResult(
        m4b=m4b,
        mp3=mp3,
        chapters=len(chapters),
        duration_seconds=chapters[-1][1],
        skipped_pages=skipped,
        cache_hits=chunk_cache.hits,
        cache_misses=chunk_cache.misses,
        total_pages=len(raw_pages),
        total_chunks=total_chunks,
        engine_version=";".join(
            e.engine_version for e in loaded_engines.values()
        ),
        details=manifest,
    )


def _uniform_rate(path: Path, work_dir: Path, target_sr: int) -> Path:
    """Return a WAV at the target sample rate, resampling only if needed."""
    if read_wav_sample_rate(path) == target_sr:
        return path
    out = work_dir / f"r{target_sr}-{path.stem}.wav"
    if not out.is_file():
        resample_wav(path, out, target_sr)
    return out


def _synthesize_with_retry(engine, text, voice_cfg, speed, lang, gap, key):
    try:
        return engine.synthesize(
            text, voice_config=voice_cfg, speed=speed, lang=lang, gap=gap
        )
    except Exception:
        logger.warning("chunk %s failed; retrying once", key[:12])
        time.sleep(0.5)
        return engine.synthesize(
            text, voice_config=voice_cfg, speed=speed, lang=lang, gap=gap
        )
