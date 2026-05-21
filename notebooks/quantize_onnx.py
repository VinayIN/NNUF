#!/usr/bin/env python3
"""
ONNX Quantization for SELD Audio-Visual Model.

Pipeline:
  1. Load PyTorch model from checkpoint
  2. Export to ONNX (float32)
  3. Apply Dynamic Quantization (INT8)
  4. Run inference on dummy data and compare outputs
  5. Log all metrics to MLflow
"""

import os
import sys
import time
import pickle
import tempfile

import mlflow
import numpy as np
import torch
import onnx
import onnxruntime

from onnxruntime.quantization import quantize_dynamic, QuantType

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from seld_pkg.DCASE2025_seld_baseline.model import SELDModel

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "seld_pkg/DCASE2025_seld_baseline/checkpoints",
    "SELDnet_audio_visual_multiACCDOA_20250331_173131",
)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "onnx_quantization"
BATCH_SIZE = 1
N_INFERENCE_ITERS = 10

# Quantization settings (matching ONNX dynamic INT8 for parity)
QUANT_MODE = "dynamic_int8"  # weights-only INT8, FP32 activations (no calibration)
IS_PER_CHANNEL = False       # per-tensor quantization (matches ONNX default)
# ─────────────────────────────────────────────────────────────────────────────


def compute_output_metrics(pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    mse = np.mean((pred_a - pred_b) ** 2)
    mae = np.mean(np.abs(pred_a - pred_b))
    max_err = np.max(np.abs(pred_a - pred_b))
    eps = 1e-10
    signal_power = np.mean(pred_a ** 2)
    noise_power = np.mean((pred_a - pred_b) ** 2)
    snr = 10.0 * np.log10(signal_power / (noise_power + eps))
    return {
        "mse": float(mse),
        "mae": float(mae),
        "max_abs_error": float(max_err),
        "snr_db": float(snr),
    }


def measure_latency(session, input_dict: dict) -> dict:
    for _ in range(5):
        session.run(None, input_dict)
    latencies = []
    for _ in range(N_INFERENCE_ITERS):
        t0 = time.perf_counter()
        session.run(None, input_dict)
        latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")

    config_path = os.path.join(MODEL_DIR, "config.pkl")
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")

    with open(config_path, "rb") as f:
        params = pickle.load(f)
    print(f"Modality: {params['modality']}")

    model = SELDModel(params).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["seld_model"])
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 ** 2)

    dummy_audio = torch.randn(BATCH_SIZE, 2, 251, 64).to(device)
    dummy_video = torch.randn(BATCH_SIZE, 50, 7, 7).to(device)

    with torch.no_grad():
        ref_output = model(dummy_audio, dummy_video).cpu().numpy()
    print(f"Output shape: {ref_output.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = os.path.join(tmpdir, "model.onnx")
        quantized_onnx_path = os.path.join(tmpdir, "model_quantized.onnx")

        print("Exporting to ONNX...")
        torch.onnx.export(
            model,
            (dummy_audio, dummy_video),
            onnx_path,
            input_names=["audio_feat", "video_feat"],
            output_names=["output"],
            dynamic_axes={
                "audio_feat": {0: "batch_size"},
                "video_feat": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
            dynamo=False,
        )
        onnx_size_bytes = os.path.getsize(onnx_path)
        print(f"ONNX FP32 size: {onnx_size_bytes / 1024 ** 2:.2f} MB")

        onnx.checker.check_model(onnx.load(onnx_path))

        print("Running FP32 inference...")
        ort_session_fp32 = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        ort_inputs = {
            "audio_feat": dummy_audio.cpu().numpy(),
            "video_feat": dummy_video.cpu().numpy(),
        }
        onnx_fp32_output = ort_session_fp32.run(None, ort_inputs)[0]

        onnx_fp32_metrics = compute_output_metrics(ref_output, onnx_fp32_output)
        onnx_fp32_latency = measure_latency(ort_session_fp32, ort_inputs)

        print(f"Applying {QUANT_MODE} Quantization (INT8)...")
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_onnx_path,
            per_channel=IS_PER_CHANNEL,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Add"],
        )
        quantized_size_bytes = os.path.getsize(quantized_onnx_path)
        print(f"Quantized ONNX size: {quantized_size_bytes / 1024 ** 2:.2f} MB")

        onnx.checker.check_model(onnx.load(quantized_onnx_path))

        print("Running quantized inference...")
        ort_session_int8 = onnxruntime.InferenceSession(
            quantized_onnx_path, providers=["CPUExecutionProvider"]
        )
        quant_output = ort_session_int8.run(None, ort_inputs)[0]

        quant_metrics = compute_output_metrics(ref_output, quant_output)
        quant_latency = measure_latency(ort_session_int8, ort_inputs)

        print("Logging to MLflow...")
        with mlflow.start_run() as run:
            mlflow.set_tag("model_type", params.get("net_type", "SELDnet"))
            mlflow.set_tag("modality", params.get("modality", "audio_visual"))
            mlflow.set_tag("quantization_type", QUANT_MODE)
            mlflow.set_tag("framework", "onnx")

            for k, v in params.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)
            mlflow.log_param("onnx_opset", 17)
            mlflow.log_param("quantization_approach", QUANT_MODE)
            mlflow.log_param("weight_type", "QInt8")

            mlflow.log_metric("original_model_size_mb", round(model_size_mb, 4))
            mlflow.log_metric("onnx_fp32_size_mb", round(onnx_size_bytes / (1024 ** 2), 4))
            mlflow.log_metric("onnx_quantized_size_mb", round(quantized_size_bytes / (1024 ** 2), 4))
            mlflow.log_metric("compression_ratio", round(onnx_size_bytes / quantized_size_bytes, 4))

            for prefix, metrics, latency in [
                ("onnx_fp32", onnx_fp32_metrics, onnx_fp32_latency),
                ("onnx_quant", quant_metrics, quant_latency),
            ]:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{prefix}_{metric_name}", value)
                for metric_name, value in latency.items():
                    mlflow.log_metric(f"{prefix}_{metric_name}", value)

            mlflow.log_artifact(onnx_path, artifact_path="models")
            mlflow.log_artifact(quantized_onnx_path, artifact_path="models")

            print(f"MLflow Run ID: {run.info.run_id}")

    print(f"\nOriginal: {model_size_mb:.2f}MB | ONNX: {onnx_size_bytes/1024**2:.2f}MB | Quantized: {quantized_size_bytes/1024**2:.2f}MB")
    print(f"Compression: {onnx_size_bytes/quantized_size_bytes:.2f}x")
    print(f"FP32  MSE: {onnx_fp32_metrics['mse']:.2e}  Latency: {onnx_fp32_latency['latency_mean_ms']:.2f}ms")
    print(f"Quant MSE: {quant_metrics['mse']:.2e}  Latency: {quant_latency['latency_mean_ms']:.2f}ms")
    print(f"MLflow: {MLFLOW_URI}")


if __name__ == "__main__":
    main()
