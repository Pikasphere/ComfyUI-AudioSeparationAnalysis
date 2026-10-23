"""Shared helpers for the ASA (AudioSeparationAnalysis) nodes.

Separation goes through pymss (MSS model zoo), transcription through ComfyUI's
own SheetSage2 audio encoder, so everything runs on the same device ComfyUI
picked for itself (HIP ``cuda:0`` on ROCm).
"""

import copy
import os

import numpy as np
import torch
import torchaudio

import folder_paths
import comfy.audio
import comfy.audio_encoders.audio_encoders
import comfy.model_management
import comfy.model_patcher
import comfy.utils
from comfy.audio_encoders.sheetsage2_abc import AbcRebuildError, events_to_abc

import pymss

AUDIO_EXTENSIONS = (".aac", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".wma",
                    ".mp4", ".flv", ".mkv", ".webm", ".mov")
DEFAULT_SAMPLE_RATE = 44100
STEM_SEPARATOR = "/"

_encoders = {}
_separators = {}
_label_index = {}


# ---------------------------------------------------------------- model catalog

def _entry_stems(entry):
    """Stem names the model is expected to return, from catalog metadata."""
    target = getattr(entry, "target_stem", "") or ""
    stems = [stem.strip() for stem in target.split(STEM_SEPARATOR) if stem.strip()]
    return stems


def model_labels():
    """Searchable ``architecture/category · name`` labels for every usable model.

    Built once per process: the catalog is static, and the UI only needs the
    string list for its combo box.
    """
    if _label_index:
        return list(_label_index)
    labels = []
    for entry in pymss.list_models(supported=True, include_user=True):
        architecture = getattr(entry, "architecture", "") or getattr(entry, "model_type", "")
        category = getattr(entry, "secondary_category", "") or ""
        tag = f"{architecture}/{category}" if category else architecture
        label = f"{tag} · {entry.name}"
        if label in _label_index:
            label = f"{tag} · {entry.name} (user)"
        _label_index[label] = entry.name
        labels.append(label)
    labels.sort()
    return labels


def resolve_model(label):
    """Map a combo label (or a bare model name) to the pymss catalog name."""
    if not label:
        models = model_labels()
        if not models:
            raise ValueError("the pymss model catalog is empty; install pymss to get the MSS model zoo")
        return _label_index[models[0]]
    return _label_index.get(label, label)


def model_summary(label):
    """Human-readable description of a catalog entry, for the info node."""
    entry = pymss.get_model_entry(resolve_model(label))
    size = getattr(entry, "size_bytes", 0) / 2**20
    return {
        "name": entry.name,
        "architecture": getattr(entry, "architecture", ""),
        "category": f"{getattr(entry, 'primary_category', '')}/{getattr(entry, 'secondary_category', '')}",
        "stems": _entry_stems(entry),
        "size_mib": round(size, 1),
    }


# ------------------------------------------------------------------- separation

def model_sample_rate(separator):
    """The rate a separation model expects; pymss configs keep it in different places."""
    config = separator.config
    for container in (config.get("audio", {}), config.get("training", {}), config.get("inference", {}), config):
        if hasattr(container, "get"):
            rate = container.get("sample_rate")
            if rate:
                return int(rate)
    return DEFAULT_SAMPLE_RATE


def load_separator(label, device, source, use_tta=False, batch_size=0):
    """Load (and cache) one MSS model; a different model or preset evicts the old one."""
    name = resolve_model(label)
    key = (name, device, source, bool(use_tta), int(batch_size or 0))
    cached = _separators.get(key)
    if cached is not None:
        return cached
    _separators.clear()
    torch.cuda.empty_cache()
    separator = pymss.MSSeparator.from_model_name(
        name, download=True, source=source, device=device, output_format="wav",
        use_tta=bool(use_tta), inference_params={"batch_size": int(batch_size) or None},
    )
    _separators[key] = separator
    return separator


def separate(waveform, sample_rate, separator, only_stems=None):
    """``(channels, samples)`` -> ``{stem: (channels, samples)}`` at the model rate."""
    rate = model_sample_rate(separator)
    wave = waveform.float().cpu()
    if wave.ndim > 2:
        wave = wave.mean(dim=0)
    if sample_rate != rate:
        wave = torchaudio.functional.resample(wave, sample_rate, rate)
    stems = separator.separate(wave.numpy(), pbar=False, stems=only_stems or None)
    if not stems:
        raise ValueError(f"{separator.model_type} returned no stems")
    return {name: torch.from_numpy(np.ascontiguousarray(data.T)).float()
            for name, data in stems.items()}, rate


def read_audio_file(path, offset=0.0, duration=0, mono=False):
    audio, rate = pymss.load_audio(str(path), sr=None, mono=mono, offset=float(offset or 0),
                                   duration=float(duration) if duration else None)
    data = np.asarray(audio, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    return torch.from_numpy(np.ascontiguousarray(data)), int(rate)


def save_wav(path, waveform, rate):
    data = waveform.detach().cpu().numpy()
    pymss.save_audio(str(path), data.T if data.ndim > 1 else data, rate, "wav", {"wav_bit_depth": "FLOAT"})


# ---------------------------------------------------------- SheetSage2 scoring

def audio_encoder_names():
    return folder_paths.get_filename_list("audio_encoders")


def load_encoder(name):
    """Load (and cache) one converted SheetSage2 file from models/audio_encoders."""
    path = folder_paths.get_full_path_or_raise("audio_encoders", name)
    signature = (os.path.getmtime(path), os.path.getsize(path))
    cached = _encoders.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    encoder = comfy.audio_encoders.audio_encoders.load_audio_encoder_from_sd(sd)
    del sd
    if not isinstance(encoder, comfy.audio_encoders.audio_encoders.SheetSage2AudioEncoder):
        raise ValueError(f"{name} is not a SheetSage2 audio encoder; convert the weights first.")
    _encoders.clear()
    _encoders[path] = (signature, encoder)
    return encoder


def drop_short_notes(events, minimum):
    """Copy of the decoded events with melody notes shorter than ``minimum`` removed."""
    pruned = []
    for event in events:
        melody = event["values"].get("melody")
        if melody:
            kept = [note for note in melody if note["end_time"] - event["time"] >= minimum]
            if len(kept) != len(melody):
                event = copy.deepcopy(event)
                event["values"]["melody"] = kept
        pruned.append(event)
    return pruned


def events_to_score(events, duration, melody_only):
    """Core ABC rebuild, retried with short grace notes dropped when they fall below the grid."""
    try:
        return events_to_abc(events, duration, melody_only=melody_only)
    except AbcRebuildError as exc:
        for minimum in (0.03, 0.06, 0.12, 0.25):
            try:
                abc = events_to_abc(drop_short_notes(events, minimum), duration, melody_only=melody_only)
            except AbcRebuildError:
                continue
            print(f"[ASA] dropped melody notes shorter than {minimum:.2f}s to fit the ABC grid ({exc})")
            return abc
        raise exc from None


def transcribe_score(encoder, waveform, melody_only):
    """Transcribe one mono waveform; retry with trailing silence when the last beat is truncated."""
    rate = encoder.model_sample_rate
    duration = waveform.shape[-1] / rate
    device = encoder.load_device
    events = encoder.model.transcribe(waveform[None].to(device))
    try:
        return events_to_score(events, duration, melody_only)
    except AbcRebuildError as exc:
        silence = waveform.new_zeros(int(2 * rate))
        events = encoder.model.transcribe(torch.cat([waveform, silence])[None].to(device))
        for extra in (0.5, 1.0, 1.5):
            try:
                return events_to_score(events, duration + extra, melody_only)
            except AbcRebuildError:
                continue
        raise exc from None


def transcribe_audio(encoder, waveform, sample_rate, melody_only):
    """Accepts ``(channels, samples)`` audio and returns ABC text."""
    comfy.model_management.load_model_gpu(encoder.patcher)
    mono = waveform.float().mean(dim=0)
    if sample_rate != encoder.model_sample_rate:
        mono = torchaudio.functional.resample(mono, sample_rate, encoder.model_sample_rate)
    return transcribe_score(encoder, mono, melody_only)
