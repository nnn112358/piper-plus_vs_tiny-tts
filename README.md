# ONNX 推論ベンチマーク比較レポート

2つの TTS モデルの ONNX Runtime 推論性能を、**同一の入力文・同一条件**で比較した結果をまとめる。

- **piperplus_onnx** —  **end-to-end fp16** ONNX（tsukuyomi-chan 6言語モデル）
- **tiny_tts_onnx** — text_encoder / duration_predictor / flow / decoder の **4分割 fp32** ONNX（間のアライメント計算は Python/NumPy）


[https://github.com/nnn112358/piper-plus_vs_tiny-tts/blob/main/tiny_tts_stft.mp4](tiny_tts_stft.mp4)


---

## 計測条件

| 項目 | 内容 |
|---|---|
| 入力文（共通） | `The weather is nice today, and I feel very relaxed.` |
| 言語 | 英語（両モデルが対応する共通言語） |
| 実行環境 | CPU（12th Gen Intel Core i9-12900H, 20 threads） |
| Execution Provider | `CPUExecutionProvider`（グラフ最適化 `ORT_ENABLE_ALL`） |
| 計測方法 | warmup 3 回 + 計測 5 回の平均（text→audio の end-to-end） |

> g2p（テキスト→音素）の所要時間は両モデルとも 1ms 未満で無視できるため、差は純粋に **ONNX モデルの forward 時間**による。

---

## 結果サマリ

| モデル | 構成 | モデルサイズ | サンプルレート | 音声長 | 推論時間 (mean) | RTF |
|---|---|---|---|---|---|---|
| **piperplus_onnx** | fp16 統合1本 | 38 MB | 22,050 Hz | 2.55 s | **46.4 ms** | **0.018** |
| **tiny_tts_onnx** | fp32 4分割 | 32 MB（合計） | 44,100 Hz | 3.54 s | **1,537 ms** | **0.434** |

- **piperplus_onnx が圧倒的に高速**。絶対推論時間で約 **33倍**、音声長で正規化した RTF でも約 **24倍** 速い。

## tiny_tts_onnx ステージ別内訳

tiny_tts の推論時間がどこで消費されているかを分解計測した結果（計 1,456 ms 時点）:

| ステージ | 時間 | 割合 | 種別 |
|---|---|---|---|
| text_encoder | 15.3 ms | 1.0% | ONNX |
| duration_predictor | 5.6 ms | 0.4% | ONNX |
| alignment（NumPy glue） | 34.8 ms | 2.4% | Python/NumPy |
| flow | 248.8 ms | 17.1% | ONNX |
| **decoder（vocoder）** | **1,151.1 ms** | **79.1%** | ONNX |

![tiny_tts stage breakdown](assets/tiny_breakdown.png)

**推論時間の約 8 割が decoder（vocoder）の forward 計算に集中**している。


