# MacWhisper CLI (stable-ts)

A Mac-friendly command-line tool to turn audio/video into subtitles using `stable-ts` (`stable_whisper`).
Supports forced alignment, SRT/VTT/TXT outputs, language-aware line wrapping, progress bars, and clear exit codes.

**Chip recommendation:** Optimized for Apple Silicon (M-series). Intel Macs are not tested.

---

## Features

- 🎤 Transcription & Translation with `stable_whisper`
- 📌 Forced alignment against a reference text (precise timestamps)
- 🧾 Outputs: `srt` (default), `vtt`, `txt`, or `all`
- ✂️ Smart wrapping: CJK by characters, others by words; configurable width
- ⚡ MPS acceleration on Apple Silicon; alignment auto-falls back to CPU for stability
- 🧭 Structured logs (`--log-level`) and progress bars (`tqdm`)
- 🧨 Exit codes: `0` success, `1` runtime error, `2` usage error, `3` dependency error

---

## Requirements

- macOS, recommended on Apple Silicon (M-series)
- Python 3.10+
- FFmpeg in your `PATH` (install via Homebrew)
- PyTorch Nightly (some stable wheels may not work for this project)
- Known-good builds (as of 2025-09-10):
  `torch==2.10.0.dev20250910`, `torchaudio==2.8.0.dev20250910`, `torchvision==0.24.0.dev20250910`

> **Why Nightly?**
> This project relies on features/fixes that may not be available in certain stable wheels. If you hit import/runtime issues with stable PyTorch, switch to the Nightly instructions below.

---

## Installation (with Conda)

> Don’t have Conda yet? Install Miniconda from the [official site](https://docs.conda.io/en/latest/miniconda.html).
>
> Don't have Homebrew yet? Install it from its [homepage](https://brew.sh/).

1.  **Create & activate a new conda environment (Python 3.10+)**
    ```bash
    conda create -n macwhisper python=3.11 -y
    conda activate macwhisper
    ```

2.  **Install FFmpeg (Homebrew)**
    ```bash
    brew install ffmpeg
    ```

3.  **Install Python packages (except torch)**
    ```bash
    pip install stable-ts pydub tqdm
    ```

4.  **Install PyTorch Nightly (required)**
    Known-good builds as of 2025-09-10:
    ```bash
    pip install --pre "torch==2.10.0.dev20250910" \
                   "torchaudio==2.8.0.dev20250910" \
                   "torchvision==0.24.0.dev20250910" \
                   --index-url https://download.pytorch.org/whl/nightly/cpu
    ```
    > **Alternative (if MPS is reported unavailable):**
    > Try the default PyTorch nightly index:
    > ```bash
    > pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
    > ```

5.  **(Optional) Verify MPS availability on Apple Silicon**
    ```python
    import torch
    print("PyTorch:", torch.__version__)
    print("MPS available:", torch.backends.mps.is_available())
    ```

---

## Quick Start

**Transcribe to SRT (default):**
```bash
python macwhisper_cli.py /path/to/input.mov --model medium --output_dir out
```

**Export all formats (SRT/VTT/TXT):**
```bash
python macwhisper_cli.py /path/to/input.mp3 --output-format all --output_dir out
```

**Set log level and disable progress bar:**
```bash
python macwhisper_cli.py /path/to/input.wav --log-level DEBUG --no-progress
```

---

## Forced Alignment

Use alignment when you already have a (mostly correct) transcript and want accurate timestamps.
Alignment requires a language code (`--language`) matching the primary spoken language in the audio.

```bash
python macwhisper_cli.py /path/to/input.mov \
  --align_from /path/to/reference.txt \
  --language ja \
  --model medium \
  --output-format all \
  --max-line-width 42 \
  --output_dir out
```

> **Note on bilingual audio:**
> For Chinese/Japanese audio, prefer aligning with the dominant spoken language. If truly bilingual, split the reference text by language and align separately, or run plain transcription first and then edit.

---

## CLI Options

| Option             | Default        | Notes                                                      |
|--------------------|----------------|------------------------------------------------------------|
| `input_file`       | —              | Audio/video file (any FFmpeg-readable format)              |
| `--model`          | `medium`       | `tiny` / `base` / `small` / `medium` / `large` / `large-v3`|
| `--task`           | `transcribe`   | Or `translate` (to English)                                |
| `--language`       | `auto`         | Required for alignment; auto-detected otherwise            |
| `--align_from`     | —              | Path to reference text for forced alignment                |
| `--output_dir`     | `.`            | Output directory                                           |
| `--output-format`  | `srt`          | `srt` / `vtt` / `txt` / `all`                              |
| `--max-line-width` | `42`           | Subtitle wrapping width                                    |
| `--device`         | `auto`         | `mps` (Apple Silicon) or `cpu`                             |
| `--no-fp16`        | `off`          | Disable fp16 (can help on some MPS setups)                 |
| `--log-level`      | `INFO`         | `DEBUG` / `INFO` / `WARNING` / `ERROR`                     |
| `--no-progress`    | `off`          | Disable progress bars                                      |

**Notes:**
- **Wrapping:** CJK lines wrap by characters; other languages wrap by words.
- **Alignment:** If running on MPS, alignment automatically switches to CPU for stability.

---

## Exit Codes

- `0` — Success
- `1` — Runtime error
- `2` — Usage error (bad args, file not found, etc.)
- `3` — Dependency error (e.g., FFmpeg missing)

**Use in scripts:**
```bash
python macwhisper_cli.py /path/to/input.mov || exit $?
```

---

## Troubleshooting

**FFmpeg not found**
Install via Homebrew and ensure it’s on your `PATH`:
```bash
brew install ffmpeg
which ffmpeg
```

**Model fails to load / OOM**
- Try a smaller model: `--model small`
- Close memory-heavy apps
- On Apple Silicon: try `--no-fp16`

**Alignment fails or looks off**
- Ensure `--language` matches the spoken language
- Provide a clean, correctly ordered reference text
- For mixed-language audio: align per language or transcribe first, then edit

**Progress bar missing**
- `tqdm` is optional; `pip install tqdm` or use `--no-progress`

**MPS not available**
- Reinstall PyTorch Nightly via the default nightly index (no `cpu` channel), or update macOS/Xcode CLTs:
  ```bash
  pip install --pre torch torchaudio torchvision -i https://download.pytorch.org/whl/nightly
  ```

---

## FAQ

**Q: Do I need `openai-whisper`?**
A: No. This project uses `stable_whisper` from `stable-ts` only. Model files remain compatible.

**Q: Why is Apple Silicon recommended?**
A: The tool is optimized for M-series (MPS acceleration). Intel Macs are not tested; performance/compatibility may vary.

**Q: Stable PyTorch fails—what now?**
A: Install PyTorch Nightly. Known-good builds (as of 2025-09-10):
`torch==2.10.0.dev20250910`, `torchaudio==2.8.0.dev20250910`, `torchvision==0.24.0.dev20250910`.

---

## License (MIT)

This project is licensed under the MIT License.

```
MIT License

Copyright (c) YEAR Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[... standard MIT text unchanged ...]
```

Replace `YEAR` and `Project Authors` as appropriate.
