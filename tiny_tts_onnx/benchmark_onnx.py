"""
Benchmark: PyTorch CPU  vs  ONNX Runtime CPU
Runs N_WARMUP warm-up then N_RUNS timed iterations.
Prints RTF (Real-Time Factor) comparison table.
"""
import time
import argparse
import tempfile
import os
import sys
import numpy as np
import torch
import soundfile as sf

# Make the package root (tiny-tts/) importable so `tiny_tts` and `infer`
# resolve regardless of the current working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

TEXT = "The weather is nice today, and I feel very relaxed."
N_WARMUP = 2
N_RUNS   = 5
_TMP_WAV = os.path.join(tempfile.gettempdir(), "_tinytts_bench.wav")


def bench_pytorch(ckpt: str, device: str = "cpu"):
    from tiny_tts.infer import load_engine, synthesize
    from tiny_tts.utils.config import SAMPLING_RATE

    model = load_engine(ckpt, device=device)
    model.eval()

    # warm-up
    for _ in range(N_WARMUP):
        synthesize(TEXT, _TMP_WAV, model, speaker="female", device=device)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        synthesize(TEXT, _TMP_WAV, model, speaker="female", device=device)
        times.append(time.perf_counter() - t0)

    return times, SAMPLING_RATE


def bench_onnx(onnx_dir: str, use_gpu: bool = False):
    from infer.infer_onnx import OnnxTinyTTS
    from tiny_tts.utils.config import SAMPLING_RATE

    engine = OnnxTinyTTS(onnx_dir=onnx_dir, use_gpu=use_gpu)

    # warm-up
    for _ in range(N_WARMUP):
        engine.speak(TEXT, output_path=_TMP_WAV)

    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        audio = engine.speak(TEXT, output_path=_TMP_WAV)
        times.append(time.perf_counter() - t0)
        audio_len = len(audio)

    sr = SAMPLING_RATE
    return times, sr, audio_len


def print_table(label, times_list, audio_secs):
    avg = np.mean(times_list)
    mn  = np.min(times_list)
    mx  = np.max(times_list)
    rtf = avg / audio_secs
    speed = audio_secs / avg
    print(f"  {label:<22} | avg {avg:.3f}s | min {mn:.3f}s | max {mx:.3f}s | RTF {rtf:.3f}x  (~{speed:.1f}x RT)")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX inference")
    parser.add_argument("--checkpoint", "-c",
                        default=os.path.join(_PKG_ROOT, "checkpoints", "G.pth"))
    parser.add_argument("--onnx-dir", "-o",
                        default=os.path.join(_HERE, "onnx"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--gpu-onnx", action="store_true",
                        help="Use ONNX CUDAExecutionProvider")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  TinyTTS Inference Benchmark")
    print(f"  Text : {TEXT}")
    print(f"  Runs : {N_WARMUP} warm-up + {N_RUNS} timed")
    print(f"{'='*65}\n")

    # ── PyTorch ─────────────────────────────────────────────────────────
    print(f"[PyTorch  {args.device.upper()}]")
    pt_times, sr = bench_pytorch(args.checkpoint, device=args.device)

    # Measure audio length from a real run
    from tiny_tts.infer import load_engine, synthesize
    model = load_engine(args.checkpoint, device=args.device)
    synthesize(TEXT, _TMP_WAV, model, speaker="female", device=args.device)
    import soundfile as sf
    audio_data, _ = sf.read(_TMP_WAV)
    audio_secs = len(audio_data) / sr
    print(f"  Audio duration: {audio_secs:.3f}s at {sr}Hz")

    # ── ONNX ─────────────────────────────────────────────────────────────
    print(f"\n[ONNX Runtime  (gpu={args.gpu_onnx})]")
    ort_times, sr2, n_samples = bench_onnx(args.onnx_dir, use_gpu=args.gpu_onnx)
    onnx_audio_secs = n_samples / sr2

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'-'*65}")
    print(f"  {'Backend':<22} | {'avg':>7} | {'min':>7} | {'max':>7} | RTF")
    print(f"{'-'*65}")
    print_table(f"PyTorch {args.device.upper()}", pt_times,  audio_secs)
    print_table("ONNX CPU",                       ort_times, onnx_audio_secs)

    speedup = np.mean(pt_times) / np.mean(ort_times)
    print(f"{'-'*65}")
    print(f"  ONNX is  {speedup:.2f}x  {'faster' if speedup > 1 else 'slower'} than PyTorch on {args.device.upper()}\n")


if __name__ == "__main__":
    main()
