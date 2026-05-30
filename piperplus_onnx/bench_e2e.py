"""tsukuyomi-chan-6lang-fp16.onnx (end-to-end 単一モデル) の推論時間ベンチ。

synth.py / bench_hello.py が使う 5-chunk fp32 パイプライン
(emb_lang/enc_p/dp/flow/dec) を 1 つの ONNX に統合した fp16 版を CPU 推論する。
g2p → 単一 ONNX (phoneme → audio) で完結。

複数の文長で warmup + N 回計測し、推論時間 / RTF を比較する。

    uv run --with piper-plus-g2p --with pyopenjtalk-plus --with onnxruntime \
            --with soundfile python bench_e2e.py
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from piper_plus_g2p import get_phonemizer
from piper_plus_g2p.encode.encoder import PiperEncoder

HERE = Path(__file__).resolve().parent
MODEL = HERE / "tsukuyomi-chan-6lang-fp16.onnx"
CONFIG_PATH = HERE / "config.json"

N_WARMUP = 2
N_RUNS = 5

TEXTS = [
    ("short", "こんにちは、世界"),
    ("medium", "こんにちは、つくよみちゃんです。今日は良い天気ですね。"),
    ("long", "こんにちは、つくよみちゃんです。今日はとても良い天気で、"
             "散歩日和ですね。お昼ご飯を食べたあとに、近くの公園まで"
             "歩いて行こうと思っています。"),
]


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
    lid = cfg["language_id_map"]["ja"]
    multi = "-".join(sorted(cfg["language_id_map"].keys()))
    phonemizer = get_phonemizer(multi)
    encoder = PiperEncoder(cfg["phoneme_id_map"])

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])

    size_mb = MODEL.stat().st_size / 1024 / 1024
    print(f"=== tsukuyomi 6lang  end-to-end fp16 ONNX  推論ベンチ (CPU) ===")
    print(f"model: {MODEL.name}  ({size_mb:.1f} MB)   warmup={N_WARMUP} runs={N_RUNS}\n")
    print(f"{'text':<7} {'n_ph':>4} {'audio(s)':>8} "
          f"{'mean(ms)':>9} {'min(ms)':>8} {'max(ms)':>8} {'±sd':>6} {'RTF':>6}")
    print("-" * 64)

    for tag, text in TEXTS:
        feeds, n = build_inputs(text, cfg, phonemizer, encoder, lid)

        for _ in range(N_WARMUP):
            sess.run(None, feeds)

        times, audio = [], None
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            audio, _dur = sess.run(None, feeds)
            times.append(time.perf_counter() - t0)

        a = audio.squeeze()
        audio_sec = a.shape[-1] / sr
        mean = statistics.mean(times)
        sd = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"{tag:<7} {n:>4} {audio_sec:>8.2f} "
              f"{mean*1000:>9.1f} {min(times)*1000:>8.1f} {max(times)*1000:>8.1f} "
              f"{sd*1000:>6.1f} {mean/audio_sec:>6.3f}")

        sf.write(str(HERE / f"out_e2e_{tag}.wav"),
                 np.clip(a * 32767, -32768, 32767).astype(np.int16), sr)

    print("\nRTF < 1.0 なら実時間より速い。WAV: out_e2e_{short,medium,long}.wav")


if __name__ == "__main__":
    main()
