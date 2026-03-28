# Custom Hindi Dataset Creation

Build your own Hindi speech dataset for fine-tuning CSM-1B. This toolkit provides multiple approaches depending on your resources and quality needs.

## Dataset Format

The training pipeline expects JSONL files where each line is a 2-turn conversation:

```json
{
  "conversation": [
    {
      "role": "0",
      "content": [
        {"type": "text", "text": "Hindi text (context)"},
        {"type": "audio", "path": [0.01, -0.02, ...]}
      ]
    },
    {
      "role": "1",
      "content": [
        {"type": "text", "text": "Hindi text (generation target)"}
      ]
    }
  ],
  "target_text": "Hindi text (generation target)"
}
```

- **Speaker 0**: Context turn with both text and audio (24kHz float32 array)
- **Speaker 1**: Target turn with text only (model learns to generate audio for this)
- Audio is stored as a JSON array of float32 samples at **24kHz**

## Approaches

### Approach A: Synthetic TTS (Fastest, Easiest)

Generate audio from Hindi text using a TTS model. Best for quickly scaling up data.

```
Hindi text corpus -> [TTS Model] -> Audio files -> [Build Script] -> JSONL
```

**Pros:** Fast, unlimited scale, consistent quality
**Cons:** Synthetic voice, model may learn to mimic the TTS rather than develop natural prosody

### Approach B: Record Your Own Voice (Best Quality)

Read Hindi sentences aloud and record via microphone. Produces the most natural data.

```
Hindi sentences -> [Your voice] -> Audio files -> [Build Script] -> JSONL
```

**Pros:** Natural prosody, your voice characteristics
**Cons:** Time-consuming (~3-5 hours for 200 samples), requires quiet environment

### Approach C: Existing Audio + Whisper Transcription

Use existing Hindi audio (podcasts, YouTube, audiobooks) and transcribe with Whisper.

```
Hindi audio files -> [Whisper ASR] -> Text transcriptions -> [Build Script] -> JSONL
```

**Pros:** Natural speech, can leverage large audio collections
**Cons:** Transcription errors, licensing concerns, variable audio quality

### Approach D: Mix All Sources

Combine all approaches for a diverse dataset. This typically gives the best results.

## Quick Start

### Setup

```bash
cd dataset/
uv pip install -r requirements.txt
```

### Approach A: Synthetic Data Pipeline

```bash
# Step 1: Collect Hindi text (uses included sample sentences)
uv run python scripts/01_collect_text.py

# Or download 500 sentences from IIT Bombay corpus:
uv run python scripts/01_collect_text.py --source iitb --count 500

# Step 2: Synthesize audio with F5-Hindi TTS
uv run python scripts/02_synthesize_audio.py

# Or use lighter MMS-TTS model:
uv run python scripts/02_synthesize_audio.py --tts mms

# Step 5: Build JSONL dataset
uv run python scripts/05_build_dataset.py --source synthesized

# Step 6: Validate
uv run python scripts/06_validate.py
```

### Approach B: Record Your Own Audio

```bash
# Step 1: (Optional) Collect text or use sample_sentences.txt
uv run python scripts/01_collect_text.py

# Step 3: Record (interactive, shows sentences one at a time)
uv run python scripts/03_record_audio.py

# Step 5: Build JSONL dataset
uv run python scripts/05_build_dataset.py --source recorded

# Step 6: Validate
uv run python scripts/06_validate.py
```

### Approach C: From Existing Audio

```bash
# Put your Hindi audio files (.wav, .mp3, .flac) in audio/raw/

# Step 4: Transcribe with Whisper
uv run python scripts/04_transcribe_audio.py --input_dir audio/raw/

# Step 5: Build JSONL dataset
uv run python scripts/05_build_dataset.py \
    --manifest audio/raw/manifest.txt \
    --audio_dir audio/raw/

# Step 6: Validate
uv run python scripts/06_validate.py
```

### Use with Training Pipeline

Once validated, copy the output to the main training data directory:

```bash
cp output/train_conversations.jsonl ../data/processed/
cp output/val_conversations.jsonl ../data/processed/

# Run training
cd ..
bash train.sh --quick          # Quick test first
bash train.sh                  # Full training
```

## Detailed Script Reference

### 01_collect_text.py — Gather Hindi Text

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `file` | Source: `file`, `dir`, `iitb`, `indicnlp` |
| `--input` | `text/sample_sentences.txt` | Input file/directory |
| `--count` | `500` | Sentences to download (for online sources) |
| `--output` | `text/collected_sentences.txt` | Output file |

**Text sources:**
- `file` — Load from a single text file (one sentence per line)
- `dir` — Load from all `.txt` files in a directory
- `iitb` — Download from [IIT Bombay Hindi-English Corpus](https://huggingface.co/datasets/cfilt/iitb-english-hindi) (Hindi side)
- `indicnlp` — Download from [AI4Bharat IndicNLP](https://ai4bharat.org/)

**Adding your own text:** Edit `text/sample_sentences.txt` or create new `.txt` files in `text/`. One sentence per line, `#` for comments.

### 02_synthesize_audio.py — TTS Audio Generation

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `text/collected_sentences.txt` | Input text file |
| `--output_dir` | `audio/synthesized/` | Output directory |
| `--tts` | `f5` | TTS backend: `f5` or `mms` |
| `--ref_audio` | `None` | Reference audio for voice cloning (F5 only) |
| `--ref_text` | `None` | Transcription of reference audio |
| `--start` | `0` | Resume from sentence index |
| `--limit` | `None` | Max sentences to process |

**TTS Models:**

| Model | Quality | Speed | Native SR | Notes |
|-------|---------|-------|-----------|-------|
| **F5-Hindi** (`SPRINGLab/F5-Hindi-24KHz`) | High | Slow | 24kHz | Supports voice cloning |
| **MMS-TTS** (`facebook/mms-tts-hin`) | Medium | Fast | 16kHz | Lightweight fallback |

**Voice cloning with F5-Hindi:** Provide a 5-15s reference WAV of the target voice:
```bash
uv run python scripts/02_synthesize_audio.py \
    --ref_audio my_voice.wav \
    --ref_text "मेरा नाम नमन है और मैं दिल्ली में रहता हूँ"
```

### 03_record_audio.py — Microphone Recording

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `text/sample_sentences.txt` | Sentences to read |
| `--output_dir` | `audio/recorded/` | Output directory |
| `--start` | `0` | Resume from sentence index |
| `--max_duration` | `15` | Max recording seconds |

**Tips for good recordings:**
- Use a quiet room with minimal echo
- Keep consistent distance from microphone (~15-30cm)
- Speak at natural pace, don't rush
- The script auto-trims silence from start/end
- You can skip or re-record any sentence

### 04_transcribe_audio.py — Whisper Transcription

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | (required) | Directory with audio files |
| `--output_dir` | same as input | Where to save manifest |
| `--whisper_model` | `medium` | Whisper model size |
| `--language` | `hi` | Language code |
| `--resample_24k` | `false` | Also save 24kHz copies |

### 05_build_dataset.py — Build JSONL

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `custom` | Source: `synthesized`, `recorded`, `custom` |
| `--manifest` | auto | Path to manifest.txt |
| `--audio_dir` | auto | Audio file directory |
| `--output_dir` | `output/` | Output directory |
| `--train_ratio` | `0.85` | Train/val split ratio |
| `--max_audio_sec` | `10` | Max audio duration |

### 06_validate.py — Dataset Validation

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `output/` | Directory with JSONL files |

**Checks performed:**
- Valid JSON on every line
- Correct conversation structure (2 turns, roles "0" and "1")
- Audio arrays are non-empty, reasonable length, not silent
- Text contains Devanagari characters
- Reports statistics (duration, text length, warnings)

## Directory Structure

```
dataset/
├── README.md                    # This guide
├── requirements.txt             # Extra dependencies
├── config.yaml                  # Configuration
├── scripts/
│   ├── 01_collect_text.py       # Gather Hindi text
│   ├── 02_synthesize_audio.py   # TTS audio synthesis
│   ├── 03_record_audio.py       # Microphone recording
│   ├── 04_transcribe_audio.py   # Whisper transcription
│   ├── 05_build_dataset.py      # Build CSM JSONL format
│   └── 06_validate.py           # Validate dataset
├── text/                        # Hindi text files
│   └── sample_sentences.txt     # Starter sentences (58 included)
├── audio/                       # Raw audio files
│   ├── synthesized/             # TTS-generated audio
│   └── recorded/                # Microphone recordings
└── output/                      # Final JSONL dataset
    ├── train_conversations.jsonl
    └── val_conversations.jsonl
```

## Scaling Guidelines

| Dataset Size | Sentences | Approx. Audio | Expected Quality |
|-------------|-----------|---------------|-----------------|
| Tiny (POC) | 50-100 | ~15 min | Pipeline validation only |
| Small | 200-500 | ~1 hr | Noticeable Hindi improvement |
| Medium | 1,000-2,000 | ~5 hrs | Good Hindi TTS quality |
| Large | 5,000+ | ~20+ hrs | Production-quality Hindi |

**Recommendations:**
- Start with **Approach A** (synthetic) to validate the pipeline quickly
- Add **Approach B** (recorded) data for natural prosody
- Use **Approach D** (mixed) for best results — even 50 real recordings + 500 synthetic samples can work well
- The FLEURS dataset from the main pipeline gives you 200 samples baseline — custom data adds diversity on top

## Troubleshooting

**F5-TTS installation fails:**
```bash
# Try installing with specific torch version
uv pip install f5-tts --no-deps
uv pip install cached-path jieba pypinyin
```

**sounddevice (recording) not working on Mac:**
```bash
# Grant terminal microphone access in System Preferences > Privacy > Microphone
# Or use: brew install portaudio && uv pip install sounddevice
```

**Whisper SSL error:**
```bash
uv pip install certifi
export SSL_CERT_FILE=$(uv run python -c "import certifi; print(certifi.where())")
```

**Audio sounds too fast/slow after resampling:**
All scripts target 24kHz. If your source audio is at a different sample rate, the scripts handle resampling automatically. If issues persist, check with:
```python
import soundfile as sf
audio, sr = sf.read("your_file.wav")
print(f"Sample rate: {sr}, Duration: {len(audio)/sr:.1f}s")
```
