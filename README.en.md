# ComfyUI-AudioSeparationAnalysis (ASA)

Atomic audio separation and analysis nodes for ComfyUI: **any MSS model → pick a stem → SheetSage2 score → MIDI**.

One node does one thing. Folders and batching are left to ComfyUI's own list/batch
machinery instead of a single do-everything node.

Verified on **AMD RX 9070 GRE / ROCm 7.14 / torch 2.12.0** and on **NVIDIA RTX 5060 Ti 16 GiB / CUDA 13.0 / torch 2.9.1**;
CPU-only works too, just slower. The code only relies on `torch.cuda`, which is HIP under ROCm and plain CUDA
under NVIDIA, so there is no vendor-specific path.

中文（默认）：[README.md](README.md)

## Nodes

| Node | In → Out | Notes |
|---|---|---|
| `ASA Load Audio` | path, sample_rate, mono, offset, duration → AUDIO, info | Decodes anything pymss/av can read (`.aac`, `.flv`, `.mkv`, …) that core `LoadAudio` cannot; relative paths resolve under ComfyUI `input/` |
| `ASA Separate (MSS)` | audio, model, device, source, only_stems, use_tta, batch_size → stems, names, bundle, sample_rate | **No fixed stem count**: 1/2/4/6/8/53-track models, denoise, dereverb all use this node |
| `ASA Stem Select` | bundle, stem → audio, name | Pick by name (case-insensitive substring) or 1-based index |
| `ASA SheetSage2 to ABC` | audio, model_name, mode → abc | Transcription only; the ABC text is also shown in the UI |
| `ASA ABC to MIDI` | abc, filename_prefix, save → saved_midi, bytes | Standard MIDI File: format 1, 480 ppq, tempo/meter/chord/section markers |
| `ASA Model Info` | model → info | Architecture, category, expected stems, size |
| `ASA Convert Model` | sheetsage2_dir, base_model_dir, output_name, dtype → status | Merge the SheetSage2 adapter with MERT-v2 into a ComfyUI audio encoder file |

`stems` is an `AUDIO` list (native batching, feed it to `SaveAudio`), `names` are the same
stem names as strings, and `bundle` is the dictionary `ASA Stem Select` consumes — a bundle
is required because two parallel list outputs would be zipped per item by ComfyUI, making
"pick one stem by name" impossible.

Typical graph:

```
ASA Load Audio → ASA Separate (MSS) → ASA Stem Select("vocals") → ASA SheetSage2 to ABC → ASA ABC to MIDI
                     └→ stems ─→ SaveAudio ×N
```

Example workflow: [example_workflows/ASA 音频分离与扒谱.json](example_workflows/ASA%20音频分离与扒谱.json)

![ASA example chain](docs/example-chain.svg)

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo> ComfyUI-AudioSeparationAnalysis
../.venv/Scripts/python.exe -m pip install pymss     # Linux: ../venv/bin/python -m pip install pymss
```

`pymss` (which pulls `pymss-core`) is **not** a ComfyUI dependency, so it has to be
installed into the same environment ComfyUI runs in. Restart ComfyUI afterwards.

## Models

Model weights are downloaded **from Hugging Face** by default (`source = huggingface`).
The `source` input also accepts `modelscope` (fast inside mainland China) and `hf-mirror`.

- Catalog: 326 usable entries from the `pymss` MSS model zoo, exposed in one searchable
  `model` combo as `architecture/category · name`.
- Cache: `~/.cache/pymss/models/<category>/<task>/<file>`; move it with `PYMSS_MODEL_DIR`
  or `model_dir=`.
- The first run of a model downloads its weights; the progress bar is pymss' own.
  Download time is not inference time — a 667 MB model takes ~2 min to fetch and ~2 s to
  load from disk afterwards.
- **Custom / 8-track models**: the catalog has no registered 8-stem model (it does have
  6-stem and a 53-stem one). Register your own checkpoint and it appears in the combo
  tagged `(user)`:

  ```bash
  pymss register --name my_8stems --model /path/8stems.ckpt --config /path/8stems.yaml
  ```

  `ASA Separate` never assumes a track count, so all eight stems (and their names) pass
  straight through.

### Coverage

| Need | Models |
|---|---|
| 6 stems, fast and stable | `htdemucs/six_stem · HTDemucs4_6stems.th` |
| 6 stems, best quality | `bs_roformer/six_stem · BS-Roformer-SW.ckpt`, `logic_bs_roformer.ckpt` |
| 2 stems (vocals/instrumental) | `bs_roformer/vocal_instrumental_dual`, `mel_band_roformer/vocal_instrumental_dual` |
| UVR5 denoise | `vr/vr_denoise · UVR-DeNoise-Lite.pth` |
| UVR5 dereverb / de-echo / de-bleed | `vr/vr_dereverb`, `vr/vr_deecho`, `vr/vr_deecho_dereverb`, `mel_band_roformer/dereverb_deecho`, `noise_bleed_control` |
| 4 stems | `htdemucs/four_stem · HTDemucs4.th`, `scnet/four_stem`, `mdx23c` |
| 6 drum parts | `mdx23c/drum_stems` |

## CLI (folder batch, no queue)

```bash
python custom_nodes/ComfyUI-AudioSeparationAnalysis/cli.py \
  --folder /music --out output/asa \
  --model "htdemucs/six_stem · HTDemucs4_6stems.th" \
  --stems vocals,other --source huggingface
```

## Benchmark

### RX 9070 GRE 12 GiB / ROCm 7.14 (20 s audio, download excluded)

| Model | Stems | Load | First | Warm | Peak VRAM | Stem names |
|---|---|---|---|---|---|---|
| `HTDemucs4_6stems.th` | 6 | 1.0 s | 3.6 s | 1.6 s | 0.79 GiB | drums/bass/other/vocals/guitar/piano |
| `UVR-DeNoise-Lite.pth` | 2 | 1.0 s | 1.3 s | 0.7 s | 1.04 GiB | Noise / No Noise |
| `UVR-DeEcho-DeReverb.pth` | 2 | 1.2 s | 5.8 s | 3.2 s | 2.74 GiB | No Reverb / Reverb |
| `dereverb-echo_mel_band_roformer` | 2 | 1.9 s | 6.5 s | 8.1 s | 3.16 GiB | dry / No dry |
| `BS-Roformer-SW.ckpt` | 6 | 2.1 s | 3.3 s | 2.9 s | 2.22 GiB | bass/drums/other/vocals/guitar/piano |
| `BS-Roformer-Resurrection-Inst.ckpt` | 2 | 1.2 s | 6.7 s | 6.2 s | 2.36 GiB | other / vocals |

Full chain on a 3:14 song (BS-Roformer-SW, vocals only): separation 9.8 s / 1.82 GiB peak,
SheetSage2 transcription 47 s, MIDI 6.5 KB (584 notes, 158 markers).

### RTX 5060 Ti 16 GiB / CUDA 13.0 / torch 2.9.1 (same 20 s clip)

Blackwell, compute capability **12.0 / sm_120**, driver 596.36, `torch 2.9.1+cu130`, bf16 matmul fine.
All six models load and return the same stem names as the ROCm table; peak VRAM tops out at **5.95 GiB**
(`UVR-DeEcho-DeReverb.pth`), the rest ≤ 2.6 GiB, so a 16 GiB card keeps room for image models.
The `engine.separate` path (resample → separate → `(channels, samples)`) returns 6 × `(2, 132300)` on both
devices. "First" includes cuDNN autotuning and the STFT plan, hence the 12.5 s → 0.58 s drop for DeEcho-DeReverb.

| Model | Stems | Load | First | Warm | Peak VRAM | Stem names |
|---|---|---|---|---|---|---|
| `HTDemucs4_6stems.th` | 6 | 0.18 s | 2.05 s | 1.04 s | 1.36 GiB | drums/bass/other/vocals/guitar/piano |
| `UVR-DeNoise-Lite.pth` | 2 | 0.25 s | 1.64 s | 0.17 s | 2.31 GiB | Noise / No Noise |
| `UVR-DeEcho-DeReverb.pth` | 2 | 0.34 s | 12.47 s | 0.58 s | 5.95 GiB | No Reverb / Reverb |
| `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` | 2 | 0.51 s | 1.37 s | 0.38 s | 2.57 GiB | dry / No dry |
| `BS-Roformer-SW.ckpt` | 6 | 0.54 s | 0.68 s | 0.55 s | 1.65 GiB | bass/drums/other/vocals/guitar/piano |
| `BS-Roformer-Resurrection-Inst.ckpt` | 2 | 0.18 s | 0.90 s | 0.80 s | 2.20 GiB | other / vocals |

### CPU only (same machine, 28 logical cores, torch default 20 threads)

| Model | Stems | Load | First | Warm | Peak RSS | Stem names |
|---|---|---|---|---|---|---|
| `HTDemucs4_6stems.th` | 6 | 0.17 s | 12.41 s | 11.56 s | 2.00 GiB | drums/bass/other/vocals/guitar/piano |
| `UVR-DeNoise-Lite.pth` | 2 | 0.09 s | 2.44 s | 0.95 s | 2.52 GiB | Noise / No Noise |
| `UVR-DeEcho-DeReverb.pth` | 2 | 0.25 s | 6.21 s | 5.84 s | 3.87 GiB | No Reverb / Reverb |
| `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` | 2 | 0.11 s | 13.03 s | 12.74 s | 5.42 GiB | dry / No dry |
| `BS-Roformer-SW.ckpt` | 6 | 0.19 s | 20.03 s | 19.85 s | 5.20 GiB | bass/drums/other/vocals/guitar/piano |
| `BS-Roformer-Resurrection-Inst.ckpt` | 2 | 0.09 s | 28.47 s | 28.74 s | 6.16 GiB | other / vocals |

### GPU vs CPU (warm)

| Model | GPU | CPU | GPU speedup |
|---|---|---|---|
| `HTDemucs4_6stems.th` | 1.04 s | 11.56 s | 11× |
| `UVR-DeNoise-Lite.pth` | 0.17 s | 0.95 s | 5.6× |
| `UVR-DeEcho-DeReverb.pth` | 0.58 s | 5.84 s | 10× |
| `dereverb-echo_mel_band_roformer_sdr_10.0169.ckpt` | 0.38 s | 12.74 s | 34× |
| `BS-Roformer-SW.ckpt` | 0.55 s | 19.85 s | 36× |
| `BS-Roformer-Resurrection-Inst.ckpt` | 0.80 s | 28.74 s | 36× |

The STFT/conv UVR models only gain 5–10×; the attention-stacked RoFormer models gain 30×+. On CPU the
slowest entry needs 29 s for 20 s of audio, so folder work really wants the GPU. The ROCm numbers above
were measured on a real song rather than this synthetic clip, so cross-column comparison is order-of-magnitude only.

Reproduce with `python custom_nodes/ComfyUI-AudioSeparationAnalysis/bench.py cuda` (or `cpu`); it writes
`results_<device>.json` and the deterministic 20 s clip to `%TEMP%\asa_bench\`. `BENCH_WARM=1` skips the
second warm run (useful on CPU).

## SheetSage2 weights (for the scoring node)

`ASA Convert Model` merges the released LoRA adapter with its MERT-v2 parent:

- `m-a-p/SheetSage2` ships only adapters (355 tensors); the parent is `m-a-p/MERT-v2-FullSong` (2.36 GiB).
- `scale = lora_alpha / lora_rank = 2.0`, `W += (lora_B @ lora_A) * scale` on the four encoder
  projections, plus `output_projection.weight = token_embedding.weight.clone()` (the tied
  weight has to be materialised).
- **bf16 caveat**: `MelFrontend` keeps `spectrogram.window`, `mel_scale.fb` and `mel_mean/mel_std`
  as explicit float32 buffers. Rounding those to bf16 silently degrades the mel spectrogram, so
  the transcription gets consistently worse (whole instrumental sections collapse to rests) —
  and because it is deterministic it looks like "bf16 is just like that". `converter.py` keeps
  them via `FP32_ONLY_PREFIXES = ("encoder.feature_extractor.",)`; with that fix bf16 and fp32
  weights produce byte-identical `.abc`/`.mid` output while loading twice as fast.

## Pitfalls

- **pymss' roformer models cannot run inside `torch.inference_mode()`** — ComfyUI executes nodes
  inside it (`execution.py:754`) and the group cache keys on `gamma._version`, which raises
  `RuntimeError: Inference tensors do not track version counter`. `ASA Separate` wraps loading
  and inference in `torch.inference_mode(False)` for this reason. Plain scripts without
  inference mode work fine, which makes this easy to misdiagnose as "AMD is unsupported".
- Switching models does not free the previous one by itself; `engine.load_separator` evicts and
  empties the cache.
- No `/manager/reboot` route in this ComfyUI build: restart the app after changing `custom_nodes`.

## License

MIT — see [LICENSE](LICENSE).
