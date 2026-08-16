# RaithaMitra — AI/ML Model Pipeline

## Overview
**RaithaMitra** (Farmer's Friend) is an intelligent, multi-modal agricultural advisory platform designed to empower Kannada-speaking farmers. This repository houses the dedicated **AI/ML Model Pipeline**, which processes farmer voice queries in Kannada, identifies their agricultural intent, retrieves tailored expert advice, detects distress indicators, and delivers structured JSON output.

> [!NOTE]
> **Repository Scope**: This repository focuses strictly on the **AI/ML Model Development and Pipeline**. The website, user interface, and frontend application are developed separately by another team member. The AI/ML pipeline will interface with the website via a structured JSON API.

---

## Planned Model Architecture Pipeline

The AI/ML workflow converts farmer speech into structured actionable results through a multi-stage pipeline:

```
[Farmer Audio Query (Kannada)]
             │
             ▼
   [Phase 1: Kannada ASR] ───────────────> (Kannada Text Transcription)
             │
             ▼
   [Phase 2: Intent Classification] ─────> (crop_disease | weather | market_price | government_scheme)
             │
             ▼
   [Phase 3: Advisory Retrieval] ────────> (Relevant Agricultural Advice & Solutions)
             │
             ▼
   [Phase 4: Hybrid Distress Detection] ─> (Acoustic + Linguistic Distress Scoring)
             │
             ▼
   [Phase 5: Structured JSON Output] ────> (API Payload for Website Frontend)
```

---

## Development Roadmap & Current Status

**Current Status**: **Phase 0 — Workspace Setup & Architecture Specification Complete**

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Repository & Environment Workspace Setup | **Completed** |
| **Phase 1** | Kannada ASR (Wav2Vec2 / IndicWav2Vec) | Planned |
| **Phase 2** | Kannada Agricultural Intent Classification (IndicBERT) | Planned |
| **Phase 3** | Agricultural Advisory Retrieval System | Planned |
| **Phase 4** | Hybrid Distress Detection (Voice + Text Multi-modal) | Planned |
| **Phase 5** | Complete Model Pipeline Integration | Planned |
| **Phase 6** | Comprehensive Evaluation & Benchmark Metrics | Planned |
| **Phase 7** | Website API Integration Layer | Planned |

*All model components will undergo thorough testing and evaluation prior to final integration with the website.*

---

## Repository Structure

```
RaithaMitra ML model/
├── model/                  # Core ML modules
│   ├── asr/                # Kannada Speech Recognition modules
│   ├── intent/             # Intent classification scripts & models
│   ├── advisory/           # Advisory knowledge base & retrieval logic
│   ├── distress/           # Voice & text distress detection algorithms
│   ├── pipeline/           # End-to-end pipeline coordinator
│   ├── utils/              # Helper utilities & audio preprocessors
│   └── config/             # Pipeline configuration files
├── dataset/                # Datasets (raw datasets ignored by git)
│   ├── raw/                # Raw audio & dataset collections
│   ├── processed/          # Preprocessed text & tokenized inputs
│   └── samples/            # Benchmark audio samples for testing
├── saved_models/           # Exported model weights & checkpoints
├── tests/                  # Unit and integration test suites
│   ├── asr/
│   ├── intent/
│   ├── advisory/
│   ├── distress/
│   └── pipeline/
├── evaluation/             # Evaluation & benchmarking scripts
├── scripts/                # Utility scripts & API server runners
├── docs/                   # Technical documentation & architecture specs
│   ├── model_architecture.md
│   └── development_plan.md
├── .gitignore              # Git ignore rules for ML environments
├── requirements.txt        # Dependency requirements specification
└── README.md               # Project documentation
```

---

## Setup & Environment (Future Phase)

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended for ASR and IndicBERT fine-tuning)

### Basic Setup Instructions
1. **Clone Repository**:
   ```bash
   git clone https://github.com/ullas123ad/RaithaMitra.git
   cd RaithaMitra
   ```

2. **Create Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies** (when released in Phase 1):
   ```bash
   pip install -r requirements.txt
   ```
