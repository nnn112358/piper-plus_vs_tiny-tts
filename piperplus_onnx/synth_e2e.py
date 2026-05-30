"""tsukuyomi-chan-6lang-fp16.onnx 単体で音声合成する。

5-chunk パイプラインを 1 つに統合した end-to-end fp16 ONNX。
g2p → 単一 ONNX (phoneme → audio) で完結する。

    uv run --with piper-plus-g2p --with pyopenjtalk-plus --with onnxruntime \
            --with soundfile python synth_e2e.py "こんにちは、つくよみちゃんです。" out.wav
"""
from __future__ import annotations

import argparse
import json
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


def synthesize(text, output_path="output.wav", *, lang="ja",
               noise_scale=None, length_scale=None, noise_w=None, verbose=True):
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    sr = cfg["audio"]["sample_rate"]
    inf = cfg["inference"]
    lid = cfg["language_id_map"][lang]
    multi = "-".join(sorted(cfg["language_id_map"].keys()))

    # g2p → phoneme ids + prosody
    phonemes, prosody_info = get_phonemizer(multi).phonemize_with_prosody(text)
    ids, prosf = PiperEncoder(cfg["phoneme_id_map"]).encode_with_prosody(phonemes, prosody_info)
    n = len(ids)
    prosody = np.zeros((1, n, 3), dtype=np.int64)
    for i, p in enumerate(prosf):
        if p is not None:
            prosody[0, i] = [p.a1, p.a2, p.a3]

    scales = np.array([
        noise_scale if noise_scale is not None else inf["noise_scale"],
        length_scale if length_scale is not None else inf["length_scale"],
        noise_w if noise_w is not None else inf["noise_w"],
    ], dtype=np.float32)

    feeds = {
        "input": np.array([ids], dtype=np.int64),
        "input_lengths": np.array([n], dtype=np.int64),
        "scales": scales,
        "lid": np.array([lid], dtype=np.int64),
        "prosody_features": prosody,
        "speaker_embedding": np.zeros((1, 256), dtype=np.float32),  # single speaker
        "speaker_embedding_mask": np.zeros((1, 1), dtype=np.int64),
    }

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    t0 = time.perf_counter()
    audio, _dur = sess.run(None, feeds)
    elapsed = (time.perf_counter() - t0) * 1000

    a = audio.squeeze()
    sf.write(output_path, np.clip(a * 32767, -32768, 32767).astype(np.int16), sr)
    audio_sec = a.shape[-1] / sr

    if verbose:
        print(f"text     : {text}")
        print(f"lang     : {lang} (lid={lid})   phonemes: {n}")
        print(f"audio    : {audio_sec:.2f}s @ {sr}Hz")
        print(f"infer    : {elapsed:.1f} ms   (RTF={elapsed/1000/audio_sec:.3f})")
        print(f"saved    : {output_path}")
    return audio_sec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="tsukuyomi 6lang end-to-end fp16 ONNX で音声合成")
    p.add_argument("text", nargs="?", default="こんにちは、つくよみちゃんです。今日は良い天気ですね。")
    p.add_argument("output", nargs="?", default="output.wav")
    p.add_argument("--lang", default="ja", help="言語 (ja/en/zh/es/fr/pt)")
    p.add_argument("--noise-scale", type=float, default=None)
    p.add_argument("--length-scale", type=float, default=None, help="発話速度 (大きいほど遅い)")
    p.add_argument("--noise-w", type=float, default=None)
    args = p.parse_args()
    synthesize(args.text, args.output, lang=args.lang,
               noise_scale=args.noise_scale, length_scale=args.length_scale, noise_w=args.noise_w)
