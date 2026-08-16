# RaithaMitra AI/ML Model Architecture

## 1. Overview
 RaithaMitra is an AI-driven agricultural advisory system for Kannada-speaking farmers. The core AI/ML component processes farmer audio queries and returns structured advisory and distress information, which is consumed by the RaithaMitra frontend/website.

### Integration Flow
```
Farmer Audio Query
       │
       ▼
┌──────────────┐
│  Kannada ASR │ (Wav2Vec2 / IndicWav2Vec)
└──────┬───────┘
       │ Transcription (Kannada Text)
       ▼
┌───────────────────────────┐
│ Agricultural Intent       │ (IndicBERT)
│ Classification            │ -> crop_disease, weather, market_price, government_scheme
└──────┬────────────────────┘
       │ Intent + Confidence
       ├───┐
       │   ▼
       │ ┌───────────────────────────┐
       │ │ Advisory Retrieval Engine │ (Sentence Transformers + Advisory DB)
       │ └─────────┬─────────────────┘
       │           │ Advisory Content
       ▼           ▼
┌───────────────────────────┐
│ Hybrid Distress Detection │ (Acoustic + Linguistic Multi-modal fusion)
└──────┬────────────────────┘
       │ Distress Score + Flag
       ▼
┌───────────────────────────┐
│ Structured JSON Result    │
└──────┬────────────────────┘
       │
       ▼
 RaithaMitra Website / API
```

---

## 2. Component Details

### 2.1 Kannada ASR (Automatic Speech Recognition)
- **Model**: Pre-trained fine-tuned Wav2Vec2 / IndicWav2Vec for Kannada.
- **Input**: Audio waveform (WAV/MP3/OGG, resampled to 16kHz mono).
- **Output**: Kannada text transcription string.
- **Location**: `model/asr/`

### 2.2 Agricultural Intent Classification
- **Model**: Fine-tuned IndicBERT transformer model for text classification.
- **Input**: Kannada transcription string from ASR.
- **Categories**:
  - `crop_disease`: Queries regarding plant health, pests, diseases, treatments.
  - `weather`: Queries regarding rainfall, temperature, climate forecasts.
  - `market_price`: Queries regarding crop prices, mandi rates, selling options.
  - `government_scheme`: Queries regarding agricultural subsidies, schemes, insurance.
- **Output**: `intent` label and `confidence` score.
- **Location**: `model/intent/`

### 2.3 Advisory Retrieval Engine
- **Model**: Multilingual sentence embeddings (e.g., `paraphrase-multilingual-MiniLM-L12-v2` or Indic sentence embeddings) with cosine similarity matching against a structured agricultural advisory knowledge base.
- **Input**: Query text (Kannada transcription) and detected intent.
- **Output**: Relevant advice text, recommended action, and reference links.
- **Location**: `model/advisory/`

### 2.4 Hybrid Distress Detection
- **Model**: Dual-branch multi-modal classifier combining:
  1. **Acoustic / Prosodic Branch**: Pitch, energy, speech rate, jitter/shimmer extracted via librosa/opensmile.
  2. **Linguistic Branch**: Emotion/sentiment keywords and crisis markers in Kannada transcript.
- **Output**: `distress_score` (float 0.0 - 1.0) and `distress_flag` (boolean).
- **Location**: `model/distress/`

### 2.5 Pipeline Coordinator
- **Module**: Integrates all components into an end-to-end execution flow.
- **Input**: Audio file path or byte stream.
- **Output**: Standardized JSON dictionary payload.
- **Location**: `model/pipeline/`

---

## 3. Data Interface Contract

### Output JSON Structure
```json
{
  "status": "success",
  "transcription": "ನನ್ನ ಟೊಮೆಟೊ ಬೆಳೆಯ ಎಲೆಗಳು ಹಳದಿಯಾಗುತ್ತಿವೆ",
  "intent": {
    "category": "crop_disease",
    "confidence": 0.94
  },
  "advisory": {
    "title": "Tomato Leaf Yellowing Advisory",
    "recommendation": "Use recommended copper oxychloride spray.",
    "relevance_score": 0.89
  },
  "distress": {
    "score": 0.12,
    "is_distress": false
  },
  "metadata": {
    "processing_time_ms": 420,
    "timestamp": "2026-08-14T22:50:00Z"
  }
}
```
