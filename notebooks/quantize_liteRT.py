#!/usr/bin/env python3
"""
LiteRT Quantization for SELD Audio-Visual Model.

Pipeline:
  1. Load PyTorch model from checkpoint
  2. Export to LiteRT (float32)
  3. Apply Dynamic Quantization (INT8)
  4. Run inference on dummy data and compare outputs
  5. Log all metrics to MLflow

Reference: https://github.com/google-ai-edge/litert-torch
"""

import os
import sys
import time
import pickle
import tempfile
from typing import Tuple

import mlflow
import numpy as np
import torch
import litert_torch

# PT2E quantization imports
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
from litert_torch.quantize.pt2e_quantizer import (
    get_symmetric_quantization_config,
    PT2EQuantizer,
)
from litert_torch.quantize.quant_config import QuantConfig

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from seld_pkg.seld_pkg.DCASE2025_seld_baseline.model import SELDModel

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "seld_pkg/seld_pkg/DCASE2025_seld_baseline/checkpoints",
    "SELDnet_audio_visual_multiACCDOA_20250331_173131",
)
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = "litert_quantization"
BATCH_SIZE = 1
N_INFERENCE_ITERS = 50

# Quantization settings (matching ONNX dynamic INT8 for parity)
QUANT_MODE = "dynamic_int8"  # weights-only INT8, FP32 activations (no calibration)
IS_PER_CHANNEL = False       # per-tensor quantization (matches ONNX default)
# ─────────────────────────────────────────────────────────────────────────────


def compute_output_metrics(pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Compute accuracy metrics between reference and quantized outputs."""
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


def measure_latency(model_fn, sample_inputs: Tuple[torch.Tensor, ...]) -> dict:
    """Measure inference latency over multiple iterations."""
    # Warmup
    for _ in range(5):
        _ = model_fn(*sample_inputs)
    
    latencies = []
    for _ in range(N_INFERENCE_ITERS):
        t0 = time.perf_counter()
        _ = model_fn(*sample_inputs)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    return {
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "latency_min_ms": float(np.min(latencies)),
        "latency_max_ms": float(np.max(latencies)),
    }


def apply_dynamic_quantization(
    model: torch.nn.Module,
    sample_args: Tuple[torch.Tensor, ...],
    is_per_channel: bool = False,
) -> torch.nn.Module:
    """
    Apply dynamic INT8 quantization (weights-only) via PT2E.
    
    This mirrors ONNX Runtime's quantize_dynamic:
    - Quantizes linear/conv weights to INT8
    - Keeps activations in FP32 (computed dynamically at runtime)
    - No calibration dataset required
    
    Reference: https://github.com/google-ai-edge/litert-torch#quantization
    """
    # Configure symmetric INT8 quantization for weights only
    pt2e_quantizer = PT2EQuantizer().set_global(
        get_symmetric_quantization_config(
            is_per_channel=is_per_channel,
            is_dynamic=True,  # Dynamic quantization: weights INT8, activations FP32
        )
    )
    
    # Export model to PT2E format
    pt2e_model = torch.export.export(model, sample_args).module()
    
    # Prepare: insert observers (for dynamic quant, observers are minimal)
    pt2e_model = prepare_pt2e(pt2e_model, pt2e_quantizer)
    
    # Convert: apply weight quantization (no calibration needed for dynamic)
    pt2e_model = convert_pt2e(pt2e_model, fold_quantize=False)
    
    return pt2e_model, pt2e_quantizer


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Quantization: {QUANT_MODE} (per_channel={IS_PER_CHANNEL})")

    config_path = os.path.join(MODEL_DIR, "config.pkl")
    checkpoint_path = os.path.join(MODEL_DIR, "best_model.pth")

    with open(config_path, "rb") as f:
        params = pickle.load(f)
    print(f"Modality: {params['modality']}")

    # Load model
    model = SELDModel(params).to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["seld_model"])
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    # Model size tracking
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 ** 2)

    # Create dummy inputs matching your model signature
    dummy_audio = torch.randn(BATCH_SIZE, 2, 251, 64).to(device)
    dummy_video = torch.randn(BATCH_SIZE, 50, 7, 7).to(device)
    sample_args = (dummy_audio, dummy_video)

    # Reference inference (FP32 PyTorch)
    with torch.no_grad():
        ref_output = model(*sample_args).cpu().numpy()
    print(f"Output shape: {ref_output.shape}")

    with tempfile.TemporaryDirectory() as tmpdir:
        litert_fp32_path = os.path.join(tmpdir, "model_float32.tflite")
        litert_quant_path = os.path.join(tmpdir, "model_quantized.tflite")

        # ── Step 1: Export/Convert FP32 model to LiteRT (baseline) ──────────
        print("Exporting FP32 model to LiteRT...")
        model_fp32 = litert_torch.convert(model.eval(), sample_args)
        model_fp32.export(litert_fp32_path)
        litert_fp32_size_bytes = os.path.getsize(litert_fp32_path)
        print(f"LiteRT FP32 size: {litert_fp32_size_bytes / 1024 ** 2:.2f} MB")

        # ── Step 2: Validate FP32 Inference ───────────────────────────────
        print("Running FP32 inference...")
        fp32_output = model_fp32(*sample_args)
        
        fp32_metrics = compute_output_metrics(ref_output, fp32_output)
        fp32_latency = measure_latency(model_fp32, sample_args)
        print(f"FP32 conversion MSE: {fp32_metrics['mse']:.2e}")

        # ── Step 3: Apply Dynamic Quantization (INT8 weights) ────────────
        print(f"Applying {QUANT_MODE} quantization...")

        quantized_model, pt2e_quantizer = apply_dynamic_quantization(
            model,
            sample_args,
            is_per_channel=IS_PER_CHANNEL,
        )
        # Convert quantized model to LiteRT format
        model_quant = litert_torch.convert(
            quantized_model,
            sample_args,
            quant_config=QuantConfig(pt2e_quantizer=pt2e_quantizer),
        )

        quantized_size_bytes = os.path.getsize(model_quant.export(litert_quant_path))
        print(f"Quantized LiteRT size: {quantized_size_bytes / 1024 ** 2:.2f} MB")

        # ── Step 4: Validate Quantized Inference ─────────────────────────
        print("Running quantized inference...")
        quant_output = model_quant(*sample_args)
        
        quant_metrics = compute_output_metrics(ref_output, quant_output)
        quant_latency = measure_latency(model_quant, sample_args)

        # ── Step 5: Log to MLflow ────────────────────────────────────────
        print("Logging to MLflow...")
        with mlflow.start_run() as run:
            mlflow.set_tag("model_type", params.get("net_type", "SELDnet"))
            mlflow.set_tag("modality", params.get("modality", "audio_visual"))
            mlflow.set_tag("quantization_type", QUANT_MODE)
            mlflow.set_tag("framework", "litert")
            mlflow.set_tag("litert_version", litert_torch.__version__)
            mlflow.set_tag("pytorch_version", torch.__version__)

            # Log model params
            for k, v in params.items():
                if isinstance(v, (str, int, float, bool)):
                    mlflow.log_param(k, v)
            
            # Quantization-specific params (exact parity with ONNX script)
            mlflow.log_param("quantization_approach", "dynamic")
            mlflow.log_param("weight_type", "QInt8")
            mlflow.log_param("is_per_channel", IS_PER_CHANNEL)

            # Model sizes (exact key names as ONNX script, framework prefix only)
            mlflow.log_metric("original_model_size_mb", round(model_size_mb, 4))
            mlflow.log_metric("litert_fp32_size_mb", round(litert_fp32_size_bytes / (1024 ** 2), 4))
            mlflow.log_metric("litert_quantized_size_mb", round(litert_quant_size_bytes / (1024 ** 2), 4))
            mlflow.log_metric("compression_ratio", round(litert_fp32_size_bytes / litert_quant_size_bytes, 4))

            # Accuracy metrics (exact structure as ONNX script)
            for prefix, metrics in [("litert_fp32", fp32_metrics), ("litert_quant", quant_metrics)]:
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{prefix}_{metric_name}", value)

            # Latency metrics (exact structure as ONNX script)
            for prefix, latency in [("litert_fp32", fp32_latency), ("litert_quant", quant_latency)]:
                for metric_name, value in latency.items():
                    mlflow.log_metric(f"{prefix}_{metric_name}", value)

            # Artifacts
            mlflow.log_artifact(litert_fp32_path, artifact_path="models")
            mlflow.log_artifact(litert_quant_path, artifact_path="models")

            print(f"MLflow Run ID: {run.info.run_id}")

    print(f"\nOriginal: {model_size_mb:.2f}MB | liteRT: {litert_fp32_size_bytes/1024**2:.2f}MB | Quantized: {litert_quant_size_bytes/1024**2:.2f}MB")
    print(f"Compression: {litert_fp32_size_bytes/litert_quant_size_bytes:.2f}x")
    print(f"FP32  MSE: {fp32_metrics['mse']:.2e}  Latency: {fp32_latency['latency_mean_ms']:.2f}ms")
    print(f"Quant MSE: {quant_metrics['mse']:.2e}  Latency: {quant_latency['latency_mean_ms']:.2f}ms")
    print(f"MLflow: {MLFLOW_URI}")


if __name__ == "__main__":
    main()