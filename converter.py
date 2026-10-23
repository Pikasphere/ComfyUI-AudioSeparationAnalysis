"""Merge the released SheetSage2 LoRA adapters into ComfyUI's audio encoder file.

The Hugging Face ``m-a-p/SheetSage2`` snapshot only ships ``adapter.*`` LoRA
weights; the MERT-v2-FullSong encoder and the decoding head live in two files.
ComfyUI wants one merged state dict whose keys match
``comfy.audio_encoders.sheetsage2.SheetSage2``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PROJECTIONS = ("query_proj", "key_proj", "value_proj", "out_proj")
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
# ``MelFrontend`` keeps the STFT window, the mel filterbank and the mel
# normalisation statistics in float32 on purpose (``cast_to_input`` only moves
# them to the waveform dtype), so they must not be rounded to bf16 here.
FP32_ONLY_PREFIXES = ("encoder.feature_extractor.",)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_of(handle, key):
    return tuple(handle.get_slice(key).get_shape())


def convert(sheetsage2_dir, base_model_dir, output_path, dtype="fp32", verify_base=False, log=print):
    """Write a ComfyUI audio encoder file and return a small statistics dict."""
    sheetsage2_dir = Path(sheetsage2_dir)
    base_model_dir = Path(base_model_dir)
    output_path = Path(output_path)
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")

    config = read_json(sheetsage2_dir / "config.json")
    adapter_file = sheetsage2_dir / "model.safetensors"
    base_file = base_model_dir / "model.safetensors"
    for path in (adapter_file, base_file):
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")

    if verify_base:
        expected = config.get("base_model_sha256")
        actual = sha256_file(base_file)
        if expected and actual != expected:
            raise ValueError(f"MERT-v2 parent checksum mismatch: {actual} != {expected}")
        log(f"base model sha256 {actual} matches {config.get('base_model_path')}")

    log(f"reading MERT-v2 parent {base_file} ({base_file.stat().st_size / 2**30:.2f} GiB)")
    base = safe_open(str(base_file), framework="pt", device="cpu")
    heads = safe_open(str(adapter_file), framework="pt", device="cpu")

    scale = float(config.get("lora_alpha", 0.0)) / float(config.get("lora_rank", 0.0))
    merged = config.get("weights_format") == "adapter"
    out = {"encoder." + key: base.get_tensor(key) for key in base.keys()}
    if merged:
        layers = sorted({int(key.split(".")[2]) for key in heads.keys() if key.startswith("adapter.layers.")})
        for layer in layers:
            for projection in PROJECTIONS:
                prefix = f"adapter.layers.{layer}.attn.{projection}"
                weight = out[f"encoder.layers.{layer}.attn.{projection}.weight"]
                delta = heads.get_tensor(f"{prefix}.lora_B.weight") @ heads.get_tensor(f"{prefix}.lora_A.weight")
                out[f"encoder.layers.{layer}.attn.{projection}.weight"] = weight + delta * scale
        log(f"merged {len(layers)} adapter layers, scale {scale:g}")

    for key in heads.keys():
        if not key.startswith("adapter."):
            out[key] = heads.get_tensor(key)
    # ``output_projection`` is tied to the token embedding in the reference
    # implementation, so the checkpoint never stores an explicit copy.
    out["output_projection.weight"] = heads.get_tensor("token_embedding.weight").clone()

    target_dtype = DTYPES[dtype]
    if target_dtype != torch.float32:
        out = {key: value if key.startswith(FP32_ONLY_PREFIXES) or not value.is_floating_point()
               else value.to(target_dtype) for key, value in out.items()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(out, str(tmp_path), metadata={"format": "pt"})
    tmp_path.replace(output_path)
    log(f"wrote {output_path} ({output_path.stat().st_size / 2**30:.2f} GiB, {len(out)} tensors)")
    return {
        "path": str(output_path),
        "tensors": len(out),
        "dtype": dtype,
        "size_gib": round(output_path.stat().st_size / 2**30, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Convert SheetSage2 HF weights to a ComfyUI audio encoder file")
    parser.add_argument("--sheetsage2-dir", required=True, help="downloaded m-a-p/SheetSage2 snapshot")
    parser.add_argument("--base-model-dir", required=True, help="downloaded m-a-p/MERT-v2-FullSong snapshot")
    parser.add_argument("--output", required=True, help="target file, usually ComfyUI/models/audio_encoders/sheetsage2.safetensors")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp32")
    parser.add_argument("--verify-base", action="store_true", help="verify the parent card weights against config.json")
    args = parser.parse_args()
    convert(args.sheetsage2_dir, args.base_model_dir, args.output,
            dtype=args.dtype, verify_base=args.verify_base, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
