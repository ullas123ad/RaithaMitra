"""
ASR Quality & Performance Evaluation Module for RaithaMitra Kannada ASR.

Calculates Word Error Rate (WER), Character Error Rate (CER), and inference latency.
"""

import time
import json
from typing import List, Dict, Any, Optional

try:
    import jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False

from model.asr.transcriber import transcribe_audio, ASRError
from model.asr.audio import AudioProcessingError


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Computes Character Error Rate (CER) using Levenshtein distance.
    Crucial for Kannada agglutinative script evaluation.
    """
    ref_chars = list(reference.strip())
    hyp_chars = list(hypothesis.strip())

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    # Levenshtein DP Matrix
    dp = [[0] * (len(hyp_chars) + 1) for _ in range(len(ref_chars) + 1)]

    for i in range(len(ref_chars) + 1):
        dp[i][0] = i
    for j in range(len(hyp_chars) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Substitution
            )

    return float(dp[len(ref_chars)][len(hyp_chars)]) / float(len(ref_chars))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Computes Word Error Rate (WER) using jiwer if available, otherwise Levenshtein on word tokens.
    """
    if HAS_JIWER:
        return float(jiwer.wer(reference, hypothesis))

    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        dp[i][0] = i
    for j in range(len(hyp_words) + 1):
        dp[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return float(dp[len(ref_words)][len(hyp_words)]) / float(len(ref_words))


def evaluate_asr_dataset(eval_samples: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Evaluates ASR model performance across a list of test audio samples.

    Args:
        eval_samples: List of dicts, each containing:
            - "audio_path": Path to test audio file
            - "reference_text": Ground truth Kannada transcription (Optional)

    Returns:
        Dict containing evaluation statistics, WER, CER, and latency metrics.
    """
    results = []
    total_audio_duration = 0.0
    total_processing_time = 0.0
    wer_scores = []
    cer_scores = []

    print(f"\n--- Starting ASR Evaluation on {len(eval_samples)} sample(s) ---")

    for idx, sample in enumerate(eval_samples, 1):
        audio_path = sample.get("audio_path")
        reference_text = sample.get("reference_text", "").strip()

        print(f"\n[Sample {idx}/{len(eval_samples)}] Audio: {audio_path}")

        try:
            start_eval = time.time()
            transcription_result = transcribe_audio(audio_path)
            eval_time = time.time() - start_eval

            hypothesis_text = transcription_result.get("text", "")
            duration = transcription_result.get("duration_seconds", 0.0)
            processing_time = transcription_result.get("processing_time_seconds", 0.0)

            total_audio_duration += duration
            total_processing_time += processing_time

            sample_res = {
                "sample_index": idx,
                "audio_path": audio_path,
                "hypothesis_text": hypothesis_text,
                "duration_seconds": duration,
                "processing_time_seconds": processing_time,
                "real_time_factor": round(processing_time / duration, 3) if duration > 0 else None
            }

            if reference_text:
                sample_wer = calculate_wer(reference_text, hypothesis_text)
                sample_cer = calculate_cer(reference_text, hypothesis_text)
                wer_scores.append(sample_wer)
                cer_scores.append(sample_cer)
                sample_res["reference_text"] = reference_text
                sample_res["wer"] = round(sample_wer, 4)
                sample_res["cer"] = round(sample_cer, 4)
                print(f"  Ref:  {reference_text}")
                print(f"  Hyp:  {hypothesis_text}")
                print(f"  WER:  {sample_wer:.2%} | CER: {sample_cer:.2%}")
            else:
                print(f"  Hyp:  {hypothesis_text}")
                print("  Note: No reference transcription provided for WER/CER calculation.")

            results.append(sample_res)

        except (AudioProcessingError, ASRError) as e:
            print(f"  Evaluation Failed for {audio_path}: {e}")
            results.append({
                "sample_index": idx,
                "audio_path": audio_path,
                "error": str(e)
            })

    summary = {
        "samples_evaluated": len(eval_samples),
        "total_audio_duration_seconds": round(total_audio_duration, 2),
        "total_processing_time_seconds": round(total_processing_time, 2),
        "average_rtf": round(total_processing_time / total_audio_duration, 3) if total_audio_duration > 0 else None,
        "mean_wer": round(sum(wer_scores) / len(wer_scores), 4) if wer_scores else None,
        "mean_cer": round(sum(cer_scores) / len(cer_scores), 4) if cer_scores else None,
        "reference_transcripts_provided": len(wer_scores) > 0,
        "sample_details": results
    }

    print("\n--- Evaluation Summary ---")
    print(f"Evaluated Samples:     {summary['samples_evaluated']}")
    print(f"Total Audio Duration:  {summary['total_audio_duration_seconds']}s")
    print(f"Total Processing Time: {summary['total_processing_time_seconds']}s")
    if summary["mean_wer"] is not None:
        print(f"Mean WER:              {summary['mean_wer']:.2%}")
        print(f"Mean CER:              {summary['mean_cer']:.2%}")
    else:
        print("Mean WER/CER:          N/A (Pending reference transcriptions from user)")
    print("--------------------------\n")

    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        eval_manifest_path = sys.argv[1]
        with open(eval_manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        evaluate_asr_dataset(manifest)
    else:
        print("Usage: python -m evaluation.eval_asr <path_to_manifest.json>")
        print("Example manifest format: [{'audio_path': 'dataset/samples/query1.wav', 'reference_text': 'ನನ್ನ ಬೆಳೆ...'}]")
