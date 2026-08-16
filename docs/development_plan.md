# RaithaMitra AI/ML Development Plan

This document outlines the step-by-step development lifecycle for the RaithaMitra AI/ML model pipeline.

---

## Phase 1 — Kannada ASR (Automatic Speech Recognition)
- **Objective**: Transcribe input Kannada audio queries into accurate Kannada text.
- **Input**: Kannada audio file (`.wav`, `.mp3`, 16kHz mono).
- **Output**: Kannada text transcription string.
- **Files Expected**:
  - `model/asr/loader.py` — Model initialization and loader scripts.
  - `model/asr/transcriber.py` — Audio preprocessing and inference pipeline.
  - `dataset/samples/sample_kannada_audio.wav` — Sample audio for testing.
  - `tests/asr/test_asr.py` — Unit tests for ASR module.
- **Testing Requirement**: Unit tests verifying model load, audio preprocessing, and non-empty Kannada text generation on test audio files.
- **Completion Criteria**: ASR module successfully converts sample Kannada audio into transcriptions with clear WER (Word Error Rate) baseline logged.

---

## Phase 2 — Intent Classification
- **Objective**: Classify transcribed Kannada text into agricultural intent categories (`crop_disease`, `weather`, `market_price`, `government_scheme`).
- **Input**: Kannada text transcription string.
- **Output**: Intent category string and confidence score float (0.0 to 1.0).
- **Files Expected**:
  - `model/intent/classifier.py` — IndicBERT classifier wrapper and inference logic.
  - `model/intent/trainer.py` — Training/fine-tuning script.
  - `dataset/processed/intent_dataset.json` — Labelled Kannada intent dataset.
  - `tests/intent/test_intent.py` — Unit tests for intent classification.
- **Testing Requirement**: Test classifier with sample phrases for each intent category and check confidence score bounds.
- **Completion Criteria**: Classifier achieves > 85% accuracy on test split across all 4 intent categories.

---

## Phase 3 — Advisory Retrieval
- **Objective**: Retrieve relevant, actionable agricultural advice based on farmer query and intent.
- **Input**: Farmer query text and intent category.
- **Output**: Structured advisory payload (title, advice text, action steps, relevance score).
- **Files Expected**:
  - `model/advisory/retriever.py` — Embedding search and retrieval engine.
  - `model/advisory/kb_builder.py` — Knowledge base indexing script.
  - `dataset/processed/advisory_kb.json` — Structured agricultural knowledge base.
  - `tests/advisory/test_advisory.py` — Unit tests for retrieval accuracy.
- **Testing Requirement**: Query tests validating top-k retrieval precision for common farming questions.
- **Completion Criteria**: Top-1 retrieval accuracy exceeds 80% on evaluation benchmark dataset.

---

## Phase 4 — Hybrid Distress Detection
- **Objective**: Detect emotional/financial distress signals from both acoustic properties of voice and text markers in the transcription.
- **Input**: Audio waveform and Kannada text transcription.
- **Output**: `distress_score` (0.0 to 1.0) and `distress_flag` (boolean).
- **Files Expected**:
  - `model/distress/acoustic_feature_extractor.py` — Audio pitch/energy/tempo feature extraction.
  - `model/distress/linguistic_detector.py` — Text key-phrase and distress marker scorer.
  - `model/distress/fusion_model.py` — Multi-modal fusion classifier.
  - `tests/distress/test_distress.py` — Unit tests for acoustic and linguistic distress features.
- **Testing Requirement**: Unit tests evaluating high-distress vs calm speech samples.
- **Completion Criteria**: Fusion model produces reliable distress scores with clear sensitivity/specificity tradeoffs documented.

---

## Phase 5 — Complete Model Pipeline
- **Objective**: Chain ASR, Intent Classification, Advisory Retrieval, and Distress Detection into a unified execution flow.
- **Input**: Audio file path or raw audio bytes.
- **Output**: Complete structured JSON result payload.
- **Files Expected**:
  - `model/pipeline/runner.py` — Pipeline orchestrator class.
  - `model/config/pipeline_config.json` — Configurable thresholds and model paths.
  - `tests/pipeline/test_pipeline.py` — End-to-end integration tests.
- **Testing Requirement**: End-to-end pipeline run from raw audio input to validated output JSON schema.
- **Completion Criteria**: End-to-end processing completes cleanly and outputs valid JSON matching data interface specs.

---

## Phase 6 — Evaluation
- **Objective**: Comprehensive benchmark evaluation of all ML pipeline components.
- **Input**: Standardized evaluation dataset (audio, ground truth transcriptions, intents, advisories, distress labels).
- **Output**: Evaluation metrics report (WER, Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Latency).
- **Files Expected**:
  - `evaluation/eval_asr.py` — ASR evaluation script.
  - `evaluation/eval_intent.py` — Intent metrics computation script.
  - `evaluation/eval_pipeline.py` — End-to-end benchmark script.
  - `docs/evaluation_results.md` — Performance report.
- **Testing Requirement**: Execution of automated evaluation benchmark scripts across dataset.
- **Completion Criteria**: Documented benchmark metrics covering accuracy, F1 scores, and latency benchmarks.

---

## Phase 7 — Website/API Integration
- **Objective**: Interface ML model pipeline with backend API / frontend website via clean JSON responses.
- **Input**: API POST request containing audio payload.
- **Output**: Standardized JSON HTTP response.
- **Files Expected**:
  - `scripts/api_server.py` — Lightweight REST/FastAPI wrapper for model pipeline.
  - `docs/api_spec.md` — API specification for website integration.
- **Testing Requirement**: API endpoint call tests with mocked frontend audio requests.
- **Completion Criteria**: REST API operational, returning verified JSON output within latency budget without modifying website codebase.
