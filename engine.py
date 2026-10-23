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
_label_name = {}
_labels = []


# ---------------------------------------------------------------- model catalog

# What the combo offers: the daily drivers, plus every model whose capability
# nothing else in the catalog covers. The other ~290 entries stay usable through
# the node's ``other_model`` field, which takes a bare name or a full label.
RECOMMENDED = (
    # 2 stems: vocals / instrumental
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
    "bs_roformer_voc_hyperacev2.ckpt",
    "BS-Roformer-Resurrection.ckpt",
    "mel_band_roformer_vocals_gabox.ckpt",
    "mel_band_roformer_vocals_fv7b_gabox.ckpt",
    "Kim_MelBandRoformer.ckpt",
    "melband_roformer_instvoc_duality_v1.ckpt",
    # 4 / 6 / 53 stems
    "mel_band_roformer_4stems_large_ver1.ckpt",
    "HTDemucs4_6stems.th",
    "BS-Roformer-SW.ckpt",
    "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "SCNet-large_starrytong_fixed.ckpt",
    # karaoke
    "mel_band_roformer_karaoke_gabox_v2.ckpt",
    "model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
    # dereverb / denoise / restoration
    "dereverb_bs_roformer_anvuew_sdr_22.5050.ckpt",
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "UVR-DeNoise.pth",
    "UVR-DeEcho-DeReverb.pth",
    "Apollo_LQ_MP3_restoration.ckpt",
    # single instruments
    "MDX23C-DrumSep-aufr33-jarredou.ckpt",
    "HTDemucs4_FT_bass.th",
    "becruily_guitar.ckpt",
    # narrow, but nothing else in the catalog does it
    "bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt",
    "model_chorus_bs_roformer_ep_267_sdr_24.1275.ckpt",
    "aspiration_mel_band_roformer_sdr_18.9845.ckpt",
    "speech_bs_roformer_AliceN_sdr_20.4440.ckpt",
    "UVR-BVE-4B_SN-44100-1.pth",
)


def _entry_stems(entry):
    """Stem names the model is expected to return, from catalog metadata."""
    target = getattr(entry, "target_stem", "") or ""
    stems = [stem.strip() for stem in target.split(STEM_SEPARATOR) if stem.strip()]
    return stems


def _label(entry):
    """``architecture/category · name`` label for one catalog entry."""
    architecture = getattr(entry, "architecture", "") or getattr(entry, "model_type", "")
    category = getattr(entry, "secondary_category", "") or ""
    tag = f"{architecture}/{category}" if category else architecture
    return f"{tag} · {entry.name}"


def _build_catalog():
    """Index every catalog label once, then keep only the visible ones."""
    user_names = {entry.name for entry in pymss.list_user_models()}
    for entry in pymss.list_models(supported=True, include_user=True):
        label = _label(entry)
        if label in _label_index:
            label = f"{label} (user)"
        _label_index[label] = entry.name
        _label_name.setdefault(entry.name, label)
    visible = [_label_name[name] for name in RECOMMENDED if name in _label_name]
    visible += [_label_name[name] for name in sorted(user_names)
                if name in _label_name and name not in RECOMMENDED]
    _labels[:] = visible


def model_labels():
    """Combo options: ``RECOMMENDED`` plus the user's own registered models.

    Built once per process. Hidden catalog entries remain usable through
    :func:`resolve_model`, which accepts a bare model name as well as a label.
    """
    if not _label_index:
        _build_catalog()
    return list(_labels)


def resolve_model(label):
    """Map a combo label (or a bare model name) to the pymss catalog name."""
    if not _label_index:
        _build_catalog()
    if not label:
        if not _labels:
            raise ValueError("the pymss model catalog is empty; install pymss to get the MSS model zoo")
        return _label_index[_labels[0]]
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
