"""ASA (AudioSeparationAnalysis) nodes: atomic audio in, separation, scoring, MIDI out.

Every node does one thing: load audio, run one MSS model, pick one stem,
transcribe one stem, export one MIDI. Batches and folders are ComfyUI's job
(list outputs), not buried inside a mega node.
"""

import time
from pathlib import Path

import folder_paths
import torch
import torchaudio
from comfy_execution.progress import get_progress_state

from . import engine
from .abc2midi import abc_to_midi
from .converter import convert

CATEGORY = "audio/ASA"
STEMS_TYPE = "ASA_STEMS"


def to_audio(waveform, sample_rate):
    """Wrap one processed item in ComfyUI's AUDIO dict shape."""
    return {"waveform": waveform[None].float(), "sample_rate": int(sample_rate)}


def split_stems(text):
    """``"vocals, drums"`` -> ``["vocals", "drums"]``; empty means every stem."""
    return [part.strip() for part in text.replace(",", " ").split() if part.strip()]


class ASALoadAudio:
    """Load any FFmpeg-readable file as AUDIO, with optional rate/mono/trim."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": ""}),
                "sample_rate": ("INT", {"default": 0, "min": 0, "max": 192000, "step": 100}),
                "mono": ("BOOLEAN", {"default": False}),
                "offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, path, sample_rate, mono, offset, duration):
        if not path.strip():
            raise ValueError("path is empty; point it at an audio file")
        target = Path(path.strip())
        if not target.is_absolute():
            target = Path(folder_paths.get_input_directory()) / target
        waveform, rate = engine.read_audio_file(target, offset=offset, duration=duration, mono=mono)
        if sample_rate and sample_rate != rate:
            waveform = torchaudio.functional.resample(waveform, rate, sample_rate)
            rate = sample_rate
        info = f"{target.name}: {tuple(waveform.shape)} @ {rate} Hz"
        print(f"[ASA] loaded {info}")
        return (to_audio(waveform, rate), info)


class ASASeparate:
    """Run one MSS model (Demucs / MDX23C / VR / SCNet / BS-RoFormer …) on AUDIO.

    The stem count is whatever the model returns: 1–8 tracks, denoise and
    dereverb models included. ``stems`` is a list for native ComfyUI batching,
    ``bundle`` lets :class:`ASAStemSelect` pick one track by name.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = engine.model_labels()
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (models or ["pymss catalog is empty"],),
                "device": (["auto", "cuda", "cpu"], {"default": "auto"}),
                "source": (["huggingface", "modelscope", "hf-mirror"], {"default": "huggingface"}),
                "only_stems": ("STRING", {"default": ""}),
                "use_tta": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
            "optional": {
                # Optional so prompts that already name a model keep validating.
                "other_model": ("STRING", {"default": "", "tooltip": "留空就用上面的 model。要用下拉里没有的模型时，填 pymss 模型名或完整标签。"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("AUDIO", "STRING", STEMS_TYPE, "INT")
    RETURN_NAMES = ("stems", "names", "bundle", "sample_rate")
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, model, device, source, only_stems, use_tta, batch_size,
            other_model="", unique_id=None):
        chosen = (other_model or "").strip() or model
        # pymss' roformer group cache keys on ``gamma._version``, which inference
        # tensors do not track, so this must run with ComfyUI's global
        # inference_mode switched off. Weights carry requires_grad=False, so no
        # autograd graph is built and the memory profile stays the same.
        with torch.inference_mode(False):
            separator = engine.load_separator(chosen, device, source, use_tta=use_tta, batch_size=batch_size)
            separator.progress_callback = _progress_reporter(unique_id)

            tracks, rate = engine.separate(audio["waveform"], audio["sample_rate"], separator,
                                           only_stems=split_stems(only_stems))
        names = list(tracks)
        print(f"[ASA] {engine.resolve_model(chosen)} -> {names} @ {rate} Hz")
        stems = [to_audio(tracks[name], rate) for name in names]
        bundle = {"stems": tracks, "sample_rate": rate}
        return (stems, names, bundle, rate)


def _progress_reporter(unique_id):
    """Throttled bridge from pymss progress callbacks to ComfyUI's progress bar."""
    if unique_id is None:
        return None
    state = get_progress_state()
    last = [0.0]

    def report(done, total, message=None):
        now = time.perf_counter()
        if now - last[0] < 0.4 and done < total:
            return
        last[0] = now
        state.update_progress(str(unique_id), float(done), float(max(total, 1)))

    return report


class ASAStemSelect:
    """Pick one stem out of a separation bundle by name (or 1-based index)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bundle": (STEMS_TYPE,),
                "stem": ("STRING", {"default": "vocals"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "name")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, bundle, stem):
        tracks = bundle["stems"]
        wanted = (stem or "").strip()
        name = wanted.lower()
        match = next((key for key in tracks if key.lower() == name), None)
        if match is None:
            match = next((key for key in tracks if name and name in key.lower()), None)
        if match is None and wanted.isdigit():
            index = int(wanted) - 1
            if 0 <= index < len(tracks):
                match = list(tracks)[index]
        if match is None:
            raise ValueError(f"stem {stem!r} is not in {sorted(tracks)}")
        return (to_audio(tracks[match], bundle["sample_rate"]), match)


class ASASheetSage2ToABC:
    """Transcribe one AUDIO stream (usually the separated vocal stem) to ABC text."""

    @classmethod
    def INPUT_TYPES(cls):
        names = engine.audio_encoder_names()
        return {
            "required": {
                "audio": ("AUDIO",),
                "model_name": (names or ["convert the SheetSage2 weights first"],),
                "mode": (["full", "melody"], {"default": "full"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("abc",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, audio, model_name, mode):
        encoder = engine.load_encoder(model_name)
        abc = engine.transcribe_audio(encoder, audio["waveform"][0], audio["sample_rate"],
                                      mode == "melody")
        return {"ui": {"text": [abc]}, "result": (abc,)}


class ASAABCToMidi:
    """Turn ABC text into a Standard MIDI File in the ComfyUI output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"default": "", "multiline": True}),
                "filename_prefix": ("STRING", {"default": "asa_midi"}),
                "save": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("saved_midi", "bytes")
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, abc, filename_prefix, save):
        if not abc.strip():
            raise ValueError("abc is empty; connect a transcription node first")
        data = abc_to_midi(abc)
        saved = ""
        if save:
            out_dir = Path(folder_paths.get_output_directory())
            out_dir.mkdir(parents=True, exist_ok=True)
            prefix = (filename_prefix or "asa_midi").strip() or "asa_midi"
            path = out_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mid"
            path.write_bytes(data)
            saved = str(path)
            print(f"[ASA] wrote {saved}")
        return {"ui": {"text": [saved or f"not saved ({len(data)} bytes)"]},
                "result": (saved, len(data))}


class ASAModelInfo:
    """Describe a catalog model: architecture, category, expected stems, size."""

    @classmethod
    def INPUT_TYPES(cls):
        models = engine.model_labels()
        return {
            "required": {"model": (models or ["pymss catalog is empty"],)},
            "optional": {"other_model": ("STRING", {"default": "", "tooltip": "留空就用上面的 model；填 pymss 模型名可查下拉里没有的模型。"})},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, model, other_model=""):
        summary = engine.model_summary((other_model or "").strip() or model)
        text = (f"{summary['name']}\narchitecture: {summary['architecture']}\n"
                f"category: {summary['category']}\n"
                f"stems: {', '.join(summary['stems']) or 'defined by the model config'}\n"
                f"size: {summary['size_mib']} MiB")
        return {"ui": {"text": [text]}, "result": (text,)}


class ASAConvertModel:
    """Merge the SheetSage2 adapter with MERT-v2 into a ComfyUI audio encoder file."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheetsage2_dir": ("STRING", {"default": "", "multiline": False}),
                "base_model_dir": ("STRING", {"default": "", "multiline": False}),
                "output_name": ("STRING", {"default": "sheetsage2_audio_encoder"}),
                "dtype": (["fp32", "bf16"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = CATEGORY
    OUTPUT_NODE = True

    def run(self, sheetsage2_dir, base_model_dir, output_name, dtype):
        if not sheetsage2_dir.strip():
            raise ValueError("sheetsage2_dir is empty; point it at the downloaded m-a-p/SheetSage2 folder.")
        if not base_model_dir.strip():
            raise ValueError("base_model_dir is empty; point it at the downloaded m-a-p/MERT-v2-FullSong folder.")
        output_name = output_name.strip() or "sheetsage2_audio_encoder"
        if not output_name.endswith(".safetensors"):
            output_name += ".safetensors"
        target = Path(folder_paths.models_dir) / "audio_encoders" / output_name
        stats = convert(sheetsage2_dir.strip(), base_model_dir.strip(), target, dtype=dtype, log=print)
        folder_paths.cache_helper.clear()
        status = f"converted {stats['tensors']} tensors -> {stats['path']} ({stats['size_gib']} GiB, {stats['dtype']})"
        print(f"[ASA] {status}")
        return (status,)


NODE_CLASS_MAPPINGS = {
    "ASALoadAudio": ASALoadAudio,
    "ASASeparate": ASASeparate,
    "ASAStemSelect": ASAStemSelect,
    "ASASheetSage2ToABC": ASASheetSage2ToABC,
    "ASAABCToMidi": ASAABCToMidi,
    "ASAModelInfo": ASAModelInfo,
    "ASAConvertModel": ASAConvertModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ASALoadAudio": "ASA Load Audio",
    "ASASeparate": "ASA Separate (MSS)",
    "ASAStemSelect": "ASA Stem Select",
    "ASASheetSage2ToABC": "ASA SheetSage2 to ABC",
    "ASAABCToMidi": "ASA ABC to MIDI",
    "ASAModelInfo": "ASA Model Info",
    "ASAConvertModel": "ASA Convert Model (SheetSage2)",
}
