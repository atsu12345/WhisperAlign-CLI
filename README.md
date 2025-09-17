# Whisper Align CLI

A command-line tool for forced alignment of text with audio using `stable-ts` (`stable_whisper`), optimized for macOS.
This tool focuses on generating accurately timestamped subtitles from a media file and a reference text.

**Chip recommendation:** Optimized for Apple Silicon (M-series). Intel Macs are not tested.

---

## Features

- 📌 **Forced Alignment**: Generate precise timestamps from a reference text.
- 🧾 **Multiple Export Formats**: Outputs `srt` (default), `vtt`, `txt`, or `all`.
- 🌏 **CJK Language Support**: Proper CJK text handling (removes spaces between characters).
- ⚡ **Apple Silicon Accelerated**: Uses MPS for acceleration and auto-falls back to CPU for alignment stability.
- 🧭 **Developer Friendly**: Structured logs (`--log-level`), progress bars (`tqdm`), and clear exit codes for scripting.

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

## Installation

> Don't have Conda yet? Install Miniconda from the [official site](https://docs.conda.io/en/latest/miniconda.html).
>
> Don't have Homebrew yet? Install it from its [homepage](https://brew.sh/).

1.  **Create & activate a new conda environment (Python 3.10+)**
    ```bash
    conda create -n whisper-align python=3.11 -y
    conda activate whisper-align
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

## Usage

### 1. Prepare Your Reference Text

For the best results, pre-format your reference `.txt` file. Each line in the file should correspond to a desired subtitle line. Long, unbroken paragraphs will result in a single, long subtitle segment, which is usually not desirable.

**Good `text.txt`:**
```
This is the first line.
And this is the second.
```

**Bad `text.txt`:**
```
This is the first line. And this is the second.
```

### 2. Run Alignment

Use alignment when you already have a transcript and want accurate timestamps. A language code (`--language`) matching the spoken language is required.

```bash
python -m macwhisper.cli /path/to/input.mov /path/to/reference.txt \
  --language ja \
  --model medium \
  --output-format all \
  --output_dir out
```

> For bilingual audio, specify the primary spoken language with `--language`. The tool can often handle segments of a secondary language, but for best results, ensure the reference text matches the audio content.

---

## CLI Options

| Option             | Default        | Notes                                                      |
|--------------------|----------------|------------------------------------------------------------|
| `input_file`       | —              | Audio/video file (any FFmpeg-readable format)              |
| `text_file`        | —              | Path to reference text file for alignment (required)       |
| `--model`          | `medium`       | `tiny` / `base` / `small` / `medium` / `large` / `large-v3`|
| `--language`       | —              | Required language code (e.g., zh, ja, en)                  |
| `--output_dir`     | `.`            | Output directory                                           |
| `--output-format`  | `srt`          | `srt` / `vtt` / `txt` / `all`                              |
| `--device`         | `auto`         | `mps` (Apple Silicon) or `cpu`                             |
| `--no-fp16`        | `off`          | Disable fp16 (can help on some MPS setups)                 |
| `--log-level`      | `INFO`         | `DEBUG` / `INFO` / `WARNING` / `ERROR`                     |
| `--no-progress`    | `off`          | Disable progress bars                                      |

**Notes:**
- **Alignment:** On MPS, alignment automatically switches to CPU for stability.
- **CJK Text Handling:** The tool automatically removes spaces between CJK characters.

---

## Exit Codes

- `0` — Success
- `1` — Runtime error
- `2` — Usage error (bad args, file not found, etc.)
- `3` — Dependency error (e.g., FFmpeg missing)

**Use in scripts:**
```bash
python -m macwhisper.cli /path/to/input.mov /path/to/reference.txt --language ja || exit $?
```

---

## Troubleshooting

**FFmpeg not found**
Install via Homebrew and ensure it's on your `PATH`:
```bash
brew install ffmpeg
which ffmpeg
```

**Model fails to load / OOM**
- Try a smaller model: `--model small`
- Close memory-heavy apps
- On Apple Silicon: try `--no-fp16`

**Alignment fails or looks off**
- Ensure `--language` matches the spoken language. 
- Provide a clean, correctly ordered reference text, preferably with one subtitle line per line in the text file.

**MPS not available**
- Reinstall PyTorch Nightly via the default nightly index (no `cpu` channel), or update macOS/Xcode CLTs:
  ```bash
  pip install --pre torch torchaudio torchvision -i https://download.pytorch.org/whl/nightly
  ```

---

## License (MIT)

This project is licensed under the MIT License.

Copyright (c) 2025 Rael
