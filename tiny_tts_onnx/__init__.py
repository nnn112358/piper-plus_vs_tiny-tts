"""ONNX inference bundle for TinyTTS.

Contains the ONNX Runtime inference engine, its benchmark, and the
exported ONNX models (under ``onnx/``).
"""
from infer.infer_onnx import OnnxTinyTTS

__all__ = ["OnnxTinyTTS"]
