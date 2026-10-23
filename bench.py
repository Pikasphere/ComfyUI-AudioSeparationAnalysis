"""Reproduce the README timing tables: bench.py cuda|cpu [model ...]

Downloads the models first, then for each one records load time, first run,
warm run, and peak memory (VRAM on CUDA, process working set on CPU) on a
deterministic 20 s stereo clip. Extra model names on the command line are
appended to the README set.
"""

import ctypes
import gc
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import wave

import numpy as np
import torch
import torchaudio

import pymss

OUT_DIR = pathlib.Path(tempfile.gettempdir()) / "asa_bench"
AUDIO_PATH = OUT_DIR / "bench_20s.wav"
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
SOURCE = "modelscope"

# The models listed in the README tables.
MODELS = (
    "HTDemucs4_6stems.th",
    "UVR-DeNoise-Lite.pth",
    "UVR-DeEcho-DeReverb.pth",
    "dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt",
    "BS-Roformer-SW.ckpt",
    "BS-Roformer-Resurrection-Inst.ckpt",
) + tuple(sys.argv[2:])

SR = 44100
SECONDS = 20
WARM_RUNS = int(os.environ.get("BENCH_WARM", "2"))


def make_audio(path):
    """20 s of deterministic pseudo-music: chords with a beat, stereo, 44.1 kHz."""
    rng = np.random.RandomState(20260101)
    t = np.arange(SR * SECONDS, dtype=np.float32) / SR
    beat = np.zeros_like(t)
    for start in np.arange(0.0, SECONDS, 0.5):
        i = int(start * SR)
        n = min(int(0.08 * SR), len(t) - i)
        env = np.exp(-np.linspace(0.0, 6.0, n)).astype(np.float32)
        beat[i:i + n] += env * rng.uniform(0.6, 1.0)
    mix = np.zeros((2, len(t)), dtype=np.float32)
    for ch, (root, shift) in enumerate(((110.0, 0.0), (164.81, 1.5))):
        for k, ratio in enumerate((1.0, 1.5, 2.0, 2.5, 3.0)):
            detune = 1.0 + shift * 0.001 * (k + 1)
            phase = rng.uniform(0, 2 * np.pi)
            mix[ch] += (0.16 / (k + 1)) * np.sin(2 * np.pi * root * ratio * detune * t + phase)
        mix[ch] += 0.25 * beat
    mix += 0.02 * rng.randn(*mix.shape).astype(np.float32)
    peak = float(np.max(np.abs(mix)))
    mix = (mix / peak * 0.85).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(2)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((mix.T * 32767.0).astype("<i2").tobytes())
    return mix


class _Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t)]


class Memory:
    """Peak working set (CPU, sampled) or peak allocated VRAM (CUDA) per model."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None
        self._peak = 0.0
        self._psapi = None
        try:
            self._psapi = ctypes.WinDLL("psapi")
            self._kernel32 = ctypes.WinDLL("kernel32")
            self._kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            self._psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters),
                                                         ctypes.c_ulong]
            self._psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        except (AttributeError, OSError):
            self._psapi = None

    def _working_set(self):
        if self._psapi is None:
            return 0
        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = self._psapi.GetProcessMemoryInfo(self._kernel32.GetCurrentProcess(),
                                             ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize if ok else 0

    def reset(self):
        if DEVICE == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return
        self._peak = float(self._working_set())
        self._stop.clear()

        def sample():
            while not self._stop.wait(0.05):
                self._peak = max(self._peak, float(self._working_set()))

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def peak(self):
        if DEVICE == "cuda":
            return torch.cuda.max_memory_allocated() / 2**30
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        return self._peak / 2**30


def model_sample_rate(separator):
    config = separator.config
    for container in (config.get("audio", {}), config.get("training", {}), config.get("inference", {}), config):
        if hasattr(container, "get"):
            rate = container.get("sample_rate")
            if rate:
                return int(rate)
    return SR


def main():
    mix = make_audio(AUDIO_PATH)
    memory = Memory()
    rows = []
    print(f"device={DEVICE} torch={torch.__version__} audio={SECONDS}s {SR}Hz stereo", flush=True)

    for name in MODELS:
        row = {"model": name, "device": DEVICE}
        try:
            start = time.perf_counter()
            pymss.download_model(name, source=SOURCE)
            row["download_s"] = round(time.perf_counter() - start, 1)

            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode(False):
                separator = pymss.MSSeparator.from_model_name(name, download=False, source=SOURCE,
                                                              device=DEVICE, output_format="wav")
            row["load_s"] = round(time.perf_counter() - start, 2)

            rate = model_sample_rate(separator)
            audio = mix if rate == SR else torchaudio.functional.resample(
                torch.from_numpy(mix), SR, rate).numpy()
            with torch.inference_mode(False):
                memory.reset()
                start = time.perf_counter()
                first = separator.separate(audio, pbar=False)
                row["first_s"] = round(time.perf_counter() - start, 2)

                start = time.perf_counter()
                warm = separator.separate(audio, pbar=False)
                row["warm1_s"] = round(time.perf_counter() - start, 2)
                for _ in range(max(WARM_RUNS - 1, 0)):
                    start = time.perf_counter()
                    warm = separator.separate(audio, pbar=False)
                    row["warm2_s"] = round(time.perf_counter() - start, 2)
                row["warm_s"] = row.get("warm2_s", row["warm1_s"])
            row["peak_gib"] = round(memory.peak(), 2)
            row["stems"] = list(first)
            row["stems_match"] = list(first) == list(warm)
            row["sample_rate"] = rate
            print(f"{name}: load {row['load_s']}s first {row['first_s']}s warm {row['warm_s']}s "
                  f"peak {row['peak_gib']}GiB stems {'/'.join(row['stems'])}", flush=True)
            del separator, first, warm
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"{name}: FAILED {row['error']}", flush=True)
        rows.append(row)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = OUT_DIR / f"results_{DEVICE}.json"
    out.write_text(json.dumps({"device": DEVICE, "torch": torch.__version__,
                               "cuda": torch.version.cuda, "audio_seconds": SECONDS,
                               "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
