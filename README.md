# ComfyUI-AudioSeparationAnalysis (ASA)

Atomic audio separation and analysis nodes for ComfyUI: **any MSS model → pick a stem → SheetSage2 score → MIDI**.

One node does one thing. Folders and batching are left to ComfyUI's own list/batch
machinery instead of a single do-everything node.

Verified on **AMD RX 9070 GRE / ROCm 7.14 / torch 2.12.0** (see the benchmark table).
The code only relies on `torch.cuda`, which is HIP under ROCm, so it also runs on NVIDIA.

中文文档：[README.zh.md](README.zh.md)

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

## Benchmark (RX 9070 GRE 12 GiB, 20 s audio, warm, download excluded)

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
