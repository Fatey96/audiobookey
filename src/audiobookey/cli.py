"""audiobookey command line interface (P1: single `convert` entrypoint)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

from . import __version__
from .audiobook import convert
from .engine import find_models_dir
from .reporter import LineReporter, RichReporter

# P6B quality tiers: the user-facing low/medium/high alias onto the existing
# draft/final engine tiers (low -> Kokoro draft; medium/high -> Chatterbox
# final). "high" additionally switches the output encode to the best bitrate
# (see audiobook.convert()'s encode_tier); chunk wavs and cache keys are
# identical for medium vs high, only the final encode differs.
QUALITY_ENGINE_ALIASES = {"low": "draft", "medium": "final", "high": "final"}
QUALITY_CHOICES = ("draft", "final", "low", "medium", "high")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audiobookey",
        description="Convert a PDF into an audiobook (m4b + mp3), fully locally.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    conv = sub.add_parser("convert", help="Convert a PDF to an audiobook.")
    conv.add_argument(
        "pdf",
        nargs="+",
        help="Path(s) to the input PDF(s); several positionals convert "
        "sequentially into one run (single progress view).",
    )
    conv.add_argument(
        "--ocr", action="store_true",
        help="OCR pages with no text layer (scanned books) before converting.",
    )
    conv.add_argument(
        "--ocr-lang", default="eng",
        help="Tesseract language passed to OCR (default: %(default)s).",
    )
    conv.add_argument(
        "--out", default="audiobook", help="Output directory (default: %(default)s)."
    )
    conv.add_argument(
        "--cache-root",
        default=None,
        help="Chunk cache root: with this set, chunks live under "
        "<cache-root>/<book-id>/chunks instead of <out>/cache, so different "
        "--out directories share one durable cache (default: <out>/cache).",
    )
    conv.add_argument(
        "--diarize",
        action="store_true",
        help="Assign a separate dialogue voice to quoted spans (narrator = "
        "--voice, dialogue = first 'dialogue*' voice in voices.json).",
    )
    conv.add_argument(
        "--voice", default="af_heart", help="Kokoro voice name (default: %(default)s)."
    )
    conv.add_argument(
        "--speed", type=float, default=1.0, help="Speech speed (default: %(default)s)."
    )
    conv.add_argument(
        "--quality",
        choices=QUALITY_CHOICES,
        default="medium",
        help="Synthesis quality tier: low = fastest draft voice, medium = "
        "natural clone-grade voice, high = natural clone-grade voice with "
        "best encode (default: %(default)s; draft/final still accepted).",
    )
    conv.add_argument(
        "--lang", default="en-us", help="Language for phonemization (default: %(default)s)."
    )
    conv.add_argument(
        "--lexicon",
        default=None,
        help="Lexicon file: 'pattern => replacement' lines (r: prefix = regex); "
        "applied per page before chunking.",
    )
    conv.add_argument(
        "--voice-map",
        default=None,
        help="JSON object mapping regex -> voice name; the first matching "
        "pattern chooses the voice for that chunk.",
    )
    conv.add_argument(
        "--gap",
        type=float,
        default=0.3,
        help="Sentence gap in seconds (draft engine; default: %(default)s).",
    )
    conv.add_argument(
        "--models-dir",
        default=None,
        help="Models directory override (default: ./models next to the repo).",
    )
    conv.add_argument(
        "--reporter",
        choices=["rich", "lines"],
        default="rich",
        help="'lines' emits machine-readable EVENT {json} progress lines "
        "(used by the web server's convert jobs).",
    )

    serve = sub.add_parser(
        "serve", help="Serve the web reader API (default port 8765)."
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    vox = sub.add_parser("voices", help="Manage named voices (data/voices.json).")
    vox_sub = vox.add_subparsers(dest="voices_cmd", required=True)

    ls = vox_sub.add_parser("list", help="List configured voices.")

    add = vox_sub.add_parser("add", help="Add a named voice.")
    add.add_argument("name")
    add.add_argument("--voice-id", default=None, help="Kokoro voice id (draft).")
    add.add_argument(
        "--clone", default=None, help="Path to a 5-15 s reference WAV (final)."
    )
    add.add_argument("--exaggeration", type=float, default=None)
    add.add_argument(
        "--quality", choices=["final", "draft"], default=None,
        help="Engine for this voice (default: final for --clone, draft for --voice-id).",
    )

    prev = vox_sub.add_parser("preview", help="Synthesize a short sample.")
    prev.add_argument("name")
    prev.add_argument("text")
    prev.add_argument("--out", default=None, help="Output wav path.")
    prev.add_argument("--speed", type=float, default=1.0)
    prev.add_argument(
        "--quality", choices=["final", "draft"], default="draft",
        help="Only used when the name is not in voices.json (draft default).",
    )
    prev.add_argument("--lang", default="en-us")
    prev.add_argument("--models-dir", default=None)
    return parser


def _load_voice_map(path: str) -> "VoiceMap":
    from .markup import VoiceMap

    return VoiceMap.from_file(path)


def _load_lexicon(path: str) -> "Lexicon":
    from .lexicon import load_lexicon

    return load_lexicon(path)


def _cmd_voices(args, console) -> int:
    from .audio import read_wav_duration, write_wav
    from .engine import find_models_dir
    from .engine_pool import make_engine
    from .voices import (
        load_voices,
        resolve_voice,
        save_voices,
        voices_path,
        voice_fingerprint,
    )

    if args.voices_cmd == "list":
        voices = load_voices()
        if not voices:
            console.print("no voices defined (data/voices.json missing or empty)")
            return 0
        for name, cfg in voices.items():
            desc = dict(cfg)
            console.print(f"{name}: {desc}")
        return 0

    if args.voices_cmd == "add":
        voices = load_voices()
        if args.voice_id and args.clone:
            console.print("[red]error:[/red] --voice-id and --clone are exclusive")
            return 2
        if not args.voice_id and not args.clone:
            console.print("[red]error:[/red] need --voice-id X or --clone W")
            return 2
        quality = args.quality or ("final" if args.clone else "draft")
        cfg: dict = {"engine": quality}
        if args.voice_id:
            cfg["voice_id"] = args.voice_id
        if args.clone:
            from pathlib import Path

            if not Path(args.clone).is_file():
                console.print(f"[red]error:[/red] clone sample not found: {args.clone}")
                return 2
            cfg["clone"] = args.clone
            if args.exaggeration is not None:
                cfg["exaggeration"] = args.exaggeration
        save_voices({**voices, args.name: cfg})
        console.print(f"saved {args.name}: {cfg}")
        return 0

    if args.voices_cmd == "preview":
        cfg = resolve_voice(args.name, args.quality or "draft")
        quality = cfg["engine"]
        if args.speed != 1.0 and quality == "final":
            console.print(
                "[red]error:[/red] speed != 1.0 is not supported by the "
                "chatterbox engine (final)"
            )
            return 2
        models_dir = find_models_dir(args.models_dir)
        engine = make_engine(quality, models_dir)
        samples, sr = engine.synthesize(
            args.text, voice_config=cfg, speed=args.speed, lang=args.lang
        )
        out = args.out or f"preview-{args.name}.wav"
        write_wav(out, samples, sr)
        console.print(
            f"preview written: {out} ({read_wav_duration(out):.1f}s, "
            f"engine={engine.engine_version})"
        )
        return 0
    return 1


def _cmd_serve(args, console) -> int:
    import uvicorn

    from .server.api import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console()

    if args.command == "voices":
        return _cmd_voices(args, console)

    if args.command == "serve":
        return _cmd_serve(args, console)

    if args.command == "convert":
        # P6B: resolve the engine tier (draft/final) from the user-facing
        # quality tier and carry the encode tier through to the final encode.
        engine_quality = QUALITY_ENGINE_ALIASES.get(args.quality, args.quality)
        encode_tier = "best" if args.quality == "high" else "standard"
        engine = None
        try:
            models_dir = find_models_dir(args.models_dir)
            reporter = (
                LineReporter()
                if args.reporter == "lines"
                else RichReporter(console)
            )
            pdfs = [Path(p) for p in args.pdf]
            # C5B: build ONE engine for the whole batch instead of paying a
            # fresh model load + canary synthesis for every book. With
            # quality=final (Chatterbox) that engine is ~2.3 GB
            # VRAM-resident for the whole run — that is fine: the
            # alternative (re-loading the model per book) is strictly
            # worse. VRAM is released explicitly once, after the batch
            # loop (see the finally block below).
            from .engine_pool import make_engine

            engine = make_engine(engine_quality, models_dir)
            with reporter as rep:
                results = []
                for index, pdf_path in enumerate(pdfs):
                    total = len(pdfs)
                    if total > 1:
                        rep.phase(f"[{index + 1}/{total}] {pdf_path.name}")
                    if args.ocr:
                        rep.phase(
                            f"OCR pass on {pdf_path.name} "
                            f"(lang={args.ocr_lang})..."
                        )
                        pdf_path = _with_ocr_text_layer(
                            pdf_path, Path(args.out), args.ocr_lang, rep
                        )
                    results.append(
                        convert(
                            pdf_path,
                            args.out,
                            voice=args.voice,
                            speed=args.speed,
                            quality=engine_quality,
                            encode_tier=encode_tier,
                            lang=args.lang,
                            models_dir=models_dir,
                            engine=engine,
                            reporter=rep,
                            lexicon=_load_lexicon(args.lexicon)
                            if args.lexicon
                            else None,
                            voice_map=_load_voice_map(args.voice_map)
                            if args.voice_map
                            else None,
                            gap=args.gap,
                            diarize=args.diarize,
                            cache_root=args.cache_root,
                        )
                    )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if args.reporter == "lines":
                print(f"EVENT {json.dumps({'error': str(exc)})}")
            console.print(f"[red]error:[/red] {exc}")
            return 2
        finally:
            # C5B: explicit VRAM release once the batch is done. `del engine`
            # drops the last strong reference so the ~2.3 GB allocation can
            # be reclaimed; torch is imported lazily inside the guard so
            # CPU-only installs never import it; and any failure here is
            # swallowed so a release problem can never mask the conversion
            # result (or the per-book error surfaced above).
            if engine is not None:
                try:
                    import torch  # lazy: CPU-only installs never need torch

                    if torch.cuda.is_available():
                        del engine
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        console.print()
        for result in results:
            console.print("[bold green]Done.[/bold green]")
            console.print(f"  m4b:      {result.m4b}")
            console.print(f"  mp3:      {result.mp3}")
            console.print(f"  chapters: {result.chapters}")
            console.print(f"  duration: {result.duration_seconds / 60:.1f} min")
            if result.skipped_pages:
                console.print(
                    f"  skipped pages (no text): "
                    f"{', '.join(map(str, result.skipped_pages))}"
                )
            cache_total = result.cache_hits + result.cache_misses
            console.print(
                f"  cache:    {result.cache_hits}/{cache_total} chunks hit "
                f"({100 * result.cache_hit_rate:.0f}%)"
            )
        return 0
    return 1


def _with_ocr_text_layer(pdf_path: Path, out_dir: Path, ocr_lang: str, rep) -> Path:
    """OCR text-less pages of a scanned PDF into a copy with a text layer.

    Raises TesseractMissingError (RuntimeError) when no tesseract binary
    exists; the caller reports the install hint.
    """
    from . import extract as extract_mod

    scanned = extract_mod.pages_without_text(pdf_path)
    if not scanned:
        return pdf_path
    rep.phase(
        f"{len(scanned)} scanned page(s) need OCR: "
        f"{', '.join(map(str, scanned))}"
    )
    return extract_mod.ensure_text_layer(
        pdf_path, out_dir / f"{pdf_path.stem}-ocr.pdf", lang=ocr_lang
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
