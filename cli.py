"""Folder batch for the ASA pack, outside the ComfyUI queue.

    python custom_nodes/ComfyUI-AudioSeparationAnalysis/cli.py \
        --folder E:/JiJiDown/Download --out output/asa --model "htdemucs/six_stem · HTDemucs4_6stems.th"

Each file is separated once and every returned stem is written as wav, so 2, 4,
6 and 8 stem models all work without a per-track flag.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import torch

COMFY_ROOT = Path(__file__).resolve().parents[2]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine


def safe_name(path):
    name = re.sub(r"[\\/:*?\"<>|]+", "_", Path(path).stem).strip(" .")
    return name or "audio"


def collect(folder, recursive=False, limit=0):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"folder does not exist: {folder}")
    pattern = folder.rglob("*") if recursive else folder.glob("*")
    found = sorted(p for p in pattern if p.is_file() and p.suffix.lower() in engine.AUDIO_EXTENSIONS)
    return found[:limit] if limit > 0 else found


def main():
    parser = argparse.ArgumentParser(description="ASA folder separation (pymss on ROCm)")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out", default=str(COMFY_ROOT / "output" / "asa"))
    parser.add_argument("--model", required=True, help="catalog label or bare model name")
    parser.add_argument("--stems", default="", help="comma separated stems to keep, empty = all")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--source", default="huggingface", choices=("modelscope", "huggingface", "hf-mirror"))
    parser.add_argument("--seconds", type=float, default=0.0, help="only the first N seconds of each file")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = collect(args.folder, recursive=args.recursive, limit=args.limit)
    if not files:
        raise ValueError(f"no audio files found in {args.folder}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    only = [part.strip() for part in args.stems.replace(",", " ").split() if part.strip()]

    with torch.inference_mode():
        separator = engine.load_separator(args.model, args.device, args.source)
        print(f"model: {engine.resolve_model(args.model)} ({len(files)} file(s))")
        for index, path in enumerate(files, 1):
            started = time.time()
            waveform, rate = engine.read_audio_file(path, duration=args.seconds)
            tracks, model_rate = engine.separate(waveform, rate, separator, only_stems=only)
            name = safe_name(path)
            for stem, data in tracks.items():
                engine.save_wav(out_dir / f"{name}_{stem}.wav", data, model_rate)
            print(f"[{index}/{len(files)}] {Path(path).name}: {list(tracks)} "
                  f"in {time.time() - started:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
