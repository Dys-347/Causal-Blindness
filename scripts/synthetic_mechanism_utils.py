import json
import os

import torch


DEFAULT_METADATA = {
    "mechanism": "linear_one_lag",
    "cause_col": "Cause_Var",
    "target_col": "Target_Var",
    "h1_response": {
        "type": "linear_last_cause",
        "raw_gain": 2.0,
    },
}


def metadata_path_for(data_path):
    root, _ = os.path.splitext(data_path)
    return root + ".meta.json"


def load_mechanism_metadata(data_path, fallback_raw_gain=2.0):
    meta_path = metadata_path_for(data_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = dict(DEFAULT_METADATA)
        metadata["h1_response"] = dict(DEFAULT_METADATA["h1_response"])
        metadata["h1_response"]["raw_gain"] = float(fallback_raw_gain)
        metadata["metadata_missing"] = True
    metadata.setdefault("cause_col", "Cause_Var")
    metadata.setdefault("target_col", "Target_Var")
    metadata.setdefault("h1_response", {"type": "linear_last_cause", "raw_gain": float(fallback_raw_gain)})
    return metadata


def build_response_context(scaler, metadata):
    response = dict(metadata.get("h1_response", {}))
    response_type = response.get("type", "linear_last_cause")
    context = {
        "metadata": metadata,
        "response_type": response_type,
        "cause_mean": float(scaler.mean_[0]),
        "cause_scale": float(scaler.scale_[0]),
        "target_mean": float(scaler.mean_[-1]),
        "target_scale": float(scaler.scale_[-1]),
    }
    if response_type == "linear_last_cause":
        raw_gain = float(response.get("raw_gain", 2.0))
        context["raw_gain"] = raw_gain
        context["causal_gain_scaled"] = raw_gain * context["cause_scale"] / context["target_scale"]
    elif response_type == "sin_last_cause":
        amplitude = float(response.get("amplitude", 1.0))
        context["amplitude"] = amplitude
        # Local standardized gain at C=0. This is only a reference scalar; the
        # exact nonlinear response is sample-dependent and computed below.
        context["causal_gain_scaled"] = amplitude * context["cause_scale"] / context["target_scale"]
    else:
        raise ValueError(f"Unsupported h1_response type: {response_type}")
    return context


def expected_h1_change(x_orig, x_variant, intervention_name, response_context):
    if not intervention_name.startswith("cause_"):
        return torch.zeros(x_orig.shape[0], device=x_orig.device)

    response_type = response_context["response_type"]
    if response_type == "linear_last_cause":
        delta_cause_std = x_variant[:, -1, 0] - x_orig[:, -1, 0]
        return response_context["causal_gain_scaled"] * delta_cause_std

    if response_type == "sin_last_cause":
        cause_mean = response_context["cause_mean"]
        cause_scale = response_context["cause_scale"]
        target_scale = response_context["target_scale"]
        amplitude = response_context["amplitude"]
        cause_orig_raw = x_orig[:, -1, 0] * cause_scale + cause_mean
        cause_variant_raw = x_variant[:, -1, 0] * cause_scale + cause_mean
        raw_change = amplitude * (torch.sin(cause_variant_raw) - torch.sin(cause_orig_raw))
        return raw_change / target_scale

    raise ValueError(f"Unsupported response type: {response_type}")

