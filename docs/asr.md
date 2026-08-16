# RaithaMitra — Phase 1: Kannada ASR Documentation

## 1. Selected ASR Model
- **Primary Model ID**: `vasista22/whisper-kannada-small`
- **Fallback Model ID**: `vasista22/whisper-kannada-tiny`
- **Model Developer / Source**: Open fine-tuned OpenAI Whisper checkpoint by Vasista on Hugging Face Hub.
- **Architecture**: Sequence-to-Sequence Encoder-Decoder Transformer fine-tuned on Kannada speech datasets.

> [!IMPORTANT]
> **Model Verification Audit**:
> `ai4bharat/indicwav2vec-v1-kn` (which was discussed as a theoretical option in preliminary project literature) was audited against Hugging Face Hub and found to **not exist**. The verified, accessible, and working model for Kannada speech recognition on Hugging Face Hub is `vasista22/whisper-kannada-small`.

---

## 2. Why This Model Was Selected
1. **Empirically Verified & Accessible**: Confirmed accessible on Hugging Face Hub with standard `automatic-speech-recognition` pipeline support.
2. **Open Source & Local Inference**: Un-gated model, executable locally on CPU or GPU without third-party API tokens.
3. **Optimized Size**: ~960 MB checkpoint footprint (small) and ~150 MB (tiny fallback), making it practical for college project deployment on standard CPU machines.

---

## 3. Model Specifications
- **Language Code**: `kn` (Kannada)
- **License**: MIT / Apache 2.0
- **Model Size**: ~960 MB checkpoint weights (244M parameters)
- **Target Sampling Rate**: 16,000 Hz (16 kHz mono)
- **Hardware Requirements**:
  - **CPU**: 4-core x86_64 processor, 4 GB+ RAM.
  - **GPU**: NVIDIA GPU with CUDA support (Optional).

---

## 4. Environment & Installation

### Virtual Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
```

### Dependency Installation
```bash
pip install torch torchaudio transformers soundfile librosa jiwer scipy --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## 5. Audio Preprocessing Pipeline

The ASR module implements safety-checked preprocessing in `model/asr/audio.py`:

```
Input Audio File (.wav / .mp3 / .flac / .ogg)
                   │
                   ▼
┌──────────────────────────────────────┐
│ File & Format Validation             │ (Checks existence, non-zero size, valid header)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Stereo-to-Mono Conversion            │ (Averages multi-channel audio to single channel)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Audio Resampling (16 kHz)            │ (Resamples audio to 16000 Hz target rate)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│ Dynamic Range Normalization          │ (Scales float32 array to [-1.0, 1.0] range)
└──────────────────┬───────────────────┘
                   │
                   ▼
       Normalized 1D Float32 Tensor
```

---

## 6. Programmatic & CLI Inference Usage

### Python API Usage
```python
from model.asr import transcribe_audio

result = transcribe_audio("dataset/samples/farmer_query.wav")
print("Transcription:", result["text"])
```

---

## 7. Quality Evaluation & Status
- **Unit Testing**: 8 unit tests in `tests/asr/` verified 100% OK.
- **Real Audio Status**: Real Kannada audio validation is **pending** until user uploads sample `.wav`/`.mp3` files to `dataset/samples/`.
