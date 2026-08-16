"""
CLI Inference Runner for RaithaMitra Kannada ASR.

Usage:
    python -m model.asr.inference path/to/kannada_audio.wav
"""

import sys
import os
import json
from model.asr.transcriber import transcribe_audio, ASRError
from model.asr.audio import AudioProcessingError


def main():
    # Configure UTF-8 encoding for Windows terminal output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if len(sys.argv) < 2:
        print("Usage: python -m model.asr.inference <path_to_audio_file>")
        print("Example: python -m model.asr.inference dataset/samples/sample_kannada.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    if not os.path.exists(audio_path):
        print(f"Error: Target audio file '{audio_path}' does not exist.")
        sys.exit(1)

    print(f"Loading and transcribing audio: {audio_path}")

    try:
        result = transcribe_audio(audio_path)
        print("\n--- Transcription Result ---")
        print(f"Language:        {result.get('language', 'kn')}")
        print(f"Transcription:   {result.get('text', '')}")
        print(f"Processing time: {result.get('processing_time_seconds', 0.0)} seconds")
        print(f"Audio duration:  {result.get('duration_seconds', 0.0)} seconds")
        print(f"Model Used:      {result.get('model', '')}")
        print(f"Device Used:     {result.get('device', 'cpu')}")
        print("----------------------------\n")
        print("Raw JSON Output:")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except (AudioProcessingError, ASRError) as e:
        print(f"\n[ASR Error]: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Unexpected Error]: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
