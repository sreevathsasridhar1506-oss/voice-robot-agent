"""
Standalone check (M1): does faster-whisper load on the GPU and actually run?

Generates a short synthetic WAV (no real speech needed — we're proving the
CUDA/cuDNN path loads and executes, not transcription accuracy) and runs it
through faster-whisper with device="cuda". Falls back to a clear message if
CUDA isn't usable, per the plan's documented fallback (use device="cpu").
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
DURATION_S = 2
WAV_PATH = "scripts/_tmp_test_tone.wav"


def _add_nvidia_dll_dirs():
    """
    faster-whisper's ctranslate2 backend needs cuBLAS/cuDNN DLLs on the loader
    search path. The pip-installed nvidia-cublas-cu12/nvidia-cudnn-cu12 wheels
    drop them under site-packages/nvidia/<pkg>/bin, which Windows' default DLL
    search order does not include.

    os.add_dll_directory() does NOT reliably fix this for ctranslate2 — cuDNN's
    DLLs have their own nested dependencies resolved via a plain LoadLibrary
    call that doesn't consistently honor AddDllDirectory-registered paths.
    Prepending PATH directly (still process-local, not the real env var) does
    work reliably and is what ctranslate2 itself falls back to searching. Must
    run before faster_whisper/ctranslate2 is imported.
    """
    if sys.platform != "win32":
        return
    nvidia_dir = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia_dir.exists():
        return
    bin_dirs = [str(p / "bin") for p in nvidia_dir.iterdir() if (p / "bin").is_dir()]
    os.environ["PATH"] = os.pathsep.join(bin_dirs) + os.pathsep + os.environ.get("PATH", "")


_add_nvidia_dll_dirs()


def make_test_wav():
    t = np.linspace(0, DURATION_S, int(SAMPLE_RATE * DURATION_S), endpoint=False)
    tone = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sf.write(WAV_PATH, tone, SAMPLE_RATE)
    return WAV_PATH


def main():
    print("Generating test WAV...")
    wav_path = make_test_wav()

    from faster_whisper import WhisperModel

    print("Loading WhisperModel(model='small', device='cuda', compute_type='float16')...")
    t0 = time.perf_counter()
    try:
        model = WhisperModel("small", device="cuda", compute_type="float16")
        print(f"Loaded on GPU in {time.perf_counter() - t0:.2f}s")
    except Exception as exc:
        print(f"GPU load FAILED: {exc}")
        print("Falling back to device='cpu' per plan's documented fallback...")
        t0 = time.perf_counter()
        model = WhisperModel("small", device="cpu", compute_type="int8")
        print(f"Loaded on CPU in {time.perf_counter() - t0:.2f}s")

    print("Running transcribe()...")
    t0 = time.perf_counter()
    segments, info = model.transcribe(wav_path, language="en")
    text = " ".join(seg.text for seg in segments)
    print(f"transcribe() ran in {time.perf_counter() - t0:.2f}s")
    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")
    print(f"Output text (expected empty/garbage — input was a tone, not speech): {text!r}")
    print("\nSUCCESS — the model loaded and ran end to end.")


if __name__ == "__main__":
    main()
