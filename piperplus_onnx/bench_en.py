"""piperplus_onnx (end-to-end fp16 ONNX) の推論時間ベンチ (英語文)。

bench_tiny.py と同じ英語文を入力し、text -> audio の end-to-end 時間を計測する。

    uv run --with piper-plus-g2p --with pyopenjtalk-plus --with onnxruntime \
            --with soundfile python bench_en.py "<text>"
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from piper_plus_g2p import get_phonemizer
from piper_plus_g2p.encode.encoder import PiperEncoder

HERE = Path(__file__).resolve().parent
MODEL = HERE / "tsukuyomi-chan-6lang-fp16.onnx"
CONFIG_PATH = HERE / "config.json"

TEXT = sys.argv[1] if len(sys.argv) > 1 else \
    "The weather is nice today, and I feel very relaxed."
LANG = "en"
N_WARMUP = 3
N_RUNS = 5


def build_inputs(text, cfg, phonemizer, encoder, lid):
    inf = cfg["inference"]
    phonemes, prosody_info = phonemizer.phonemize_with_prosody(text)
    ids, prosf = encoder.encode_with_prosody(phonemes, prosody_info)
    n = len(ids)
    prosody = np.zeros((1, n, 3), dtype=np.int64)
    for i, p in enumerate(prosf):
        if p is not None:
            prosody[0, i] = [p.a1, p.a2, p.a3]
    return {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([n], dtype=np.int64),
        "scales": np.array(
            [inf["noise_scale"], inf["length_scale"], inf["noise_w"]], dtype=np.float32
        ),
        "lid": np.array([lid], dtype=np.int64),
        "prosody_features": prosody,
        "speaker_embedding": np.zeros((1, 256), dtype=np.float32),
        "speaker_embedding_mask": np.zeros((1, 1), dtype=np.int64),
    }, n


def main():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sr = cfg["audio"]["sample_rate"]
    lid = cfg["language_id_map"][LANG]
    multi = "-".join(sorted(cfg["language_id_map"].keys()))
    phonemizer = get_phonemizer(multi)
    encoder = PiperEncoder(cfg["phoneme_id_map"])

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])

    # --- full end-to-end (g2p + ONNX) を計測 ---
    for _ in range(N_WARMUP):
        feeds, n = build_inputs(TEXT, cfg, phonemizer, encoder, lid)
        sess.run(None, feeds)

    times, audio, n = [], None, 0
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        feeds, n = build_inputs(TEXT, cfg, phonemizer, encoder, lid)
        audio, _dur = sess.run(None, feeds)
        times.append(time.perf_counter() - t0)

    # 参考: ONNX forward 単体
    onnx_times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        onnx_times.append(time.perf_counter() - t0)

    a = audio.squeeze()
    audio_sec = a.shape[-1] / sr
    mean = statistics.mean(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0.0
    print("=== piperplus_onnx (end-to-end fp16 ONNX) ===")
    print(f"text       : {TEXT}")
    print(f"sample_rate: {sr} Hz   warmup={N_WARMUP} runs={N_RUNS}   n_phonemes={n}")
    print(f"audio      : {audio_sec:.2f} s")
    print(f"infer (e2e): mean {mean*1000:.1f} ms  "
          f"(min {min(times)*1000:.1f} / max {max(times)*1000:.1f} / "
          f"sd {sd*1000:.1f})   [g2p + ONNX]")
    print(f"infer (onnx): mean {statistics.mean(onnx_times)*1000:.1f} ms   [ONNX forward only]")
    print(f"RTF (e2e)  : {mean/audio_sec:.3f}")


if __name__ == "__main__":
    main()
