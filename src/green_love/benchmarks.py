"""
GPU benchmark matching and Crusoe Cloud speed/cost estimation.

Uses embedded Lambda Labs benchmark data to estimate speedup ratios
between local GPU and Crusoe Cloud GPUs. When the local GPU is NOT
in the benchmark table, falls back to a TFLOPS + memory bandwidth
weighted ratio (70% compute, 30% bandwidth).

Best Value = total_cost / hours_saved  (cheapest per hour of time saved).
Only on-demand pricing is used.
Cloud GPU power efficiency is estimated from local GPU efficiency.
"""

import json
import math
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"

BENCHMARK_TASKS = [
    "bert_base_squad", "bert_large_squad", "gnmt",
    "resnet50", "tacotron2", "waveglow",
]

# Maps common NVML GPU names (substrings) to benchmark keys.
# ONLY GPUs that actually appear in the Lambda Labs benchmark data.
_GPU_NAME_MAPPING = {
    "RTX 4090": "RTX 4090",
    "RTX 3090": "RTX 3090",
    "A100-SXM4-80GB": "A100 80GB SXM4",
    "A100-SXM4-40GB": "A100 40GB PCIe",
    "A100-PCIE-80GB": "A100 80GB PCIe",
    "A100-PCIE-40GB": "A100 40GB PCIe",
    "A100 80GB": "A100 80GB SXM4",
    "A100 40GB": "A100 40GB PCIe",
    "A100": "A100 80GB SXM4",
    "A10G": "A10",
    "A10": "A10",
    "A40": "RTX A6000",
    "RTX A6000": "RTX A6000",
    "RTX A5000": "RTX A6000",
    "RTX 6000 Ada": "RTX 6000 Ada",
    "L40S": "RTX 6000 Ada",
    "L40": "RTX 6000 Ada",
    "H100-SXM": "H100 80GB SXM5",
    "H100 SXM": "H100 80GB SXM5",
    "H100-PCIE": "H100 80GB PCIe",
    "H100 PCIe": "H100 80GB PCIe",
    "H100": "H100 80GB SXM5",
    "H200": "H100 80GB SXM5",
    "GH200": "GH200 96GB",
    "B200": "B200 192GB SXM5",
    "B100": "H100 80GB SXM5",
    "V100-SXM2": "V100 16GB",
    "V100-PCIE": "V100 16GB",
    "V100S": "V100 16GB",
    "V100": "V100 16GB",
    "Quadro RTX 8000": "Quadro RTX 8000",
    "Quadro RTX 6000": "Quadro RTX 8000",
    "Tesla T4": "A10",
    "T4": "A10",
}

# Typical TDPs for cloud GPUs (watts) — used for CO2/energy estimation
_CLOUD_GPU_TDP = {
    "H100 80GB SXM5": 700,
    "H100 80GB PCIe": 350,
    "A100 80GB SXM4": 400,
    "A100 80GB PCIe": 300,
    "A100 40GB PCIe": 250,
    "RTX 6000 Ada": 300,
    "RTX A6000": 300,
    "RTX 4090": 450,
    "RTX 3090": 350,
    "V100 16GB": 300,
    "A10": 150,
    "Quadro RTX 8000": 260,
    "GH200 96GB": 900,
    "B200 192GB SXM5": 1000,
}

# ── GPU Specs Database (TFLOPS + bandwidth) ──────────────────────────────────
# Used for fallback speedup estimation when local GPU is NOT in benchmark table.
# Values: (fp16_tflops, memory_bandwidth_tbps, tdp_watts)

@dataclass
class GPUSpec:
    """GPU compute/memory specs for TFLOPS-based estimation."""
    fp16_tflops: float
    memory_bandwidth_tbps: float
    tdp_watts: float

_GPU_SPECS: Dict[str, GPUSpec] = {
    # Datacenter
    "A100 80GB SXM4":   GPUSpec(312.0, 2.039, 400),
    "A100 80GB PCIe":   GPUSpec(312.0, 2.039, 300),
    "A100 40GB PCIe":   GPUSpec(312.0, 1.555, 250),
    "A100 SXM":         GPUSpec(312.0, 2.039, 400),
    "A100 PCIe":        GPUSpec(312.0, 2.039, 300),
    "A100":             GPUSpec(312.0, 2.039, 400),
    "H100 80GB SXM5":   GPUSpec(989.4, 3.35,  700),
    "H100 80GB PCIe":   GPUSpec(756.0, 2.0,   350),
    "H100 SXM":         GPUSpec(989.4, 3.35,  700),
    "H100 PCIe":        GPUSpec(756.0, 2.0,   350),
    "H100":             GPUSpec(989.4, 3.35,  700),
    "V100 16GB":        GPUSpec(125.0, 0.9,   300),
    "A10":              GPUSpec(125.0, 0.6,   150),
    "RTX A6000":        GPUSpec(155.0, 0.768, 300),
    "RTX 6000 Ada":     GPUSpec(366.7, 0.960, 300),
    "Quadro RTX 8000": GPUSpec(32.6,  0.672, 260),
    "Quadro RTX 6000": GPUSpec(32.6,  0.672, 260),
    "H200":             GPUSpec(989.4, 4.8,  700),
    "L40S":             GPUSpec(366.0, 0.864, 350),
    "L40":              GPUSpec(181.0, 0.864, 300),
    "A40":              GPUSpec(150.0, 0.696, 300),
    "A10G":             GPUSpec(125.0, 0.6,   150),
    "Tesla T4":         GPUSpec(65.0,  0.320, 70),
    "T4":               GPUSpec(65.0,  0.320, 70),
    "GH200 96GB":       GPUSpec(989.4, 4.8,  900),
    "B200 192GB SXM5":  GPUSpec(2250.0, 8.0, 1000),
    # Consumer RTX 40 series
    "RTX 4090":         GPUSpec(330.3, 1.008, 450),
    "RTX 4080 SUPER":   GPUSpec(206.1, 0.736, 320),
    "RTX 4080":         GPUSpec(206.1, 0.716, 320),
    "RTX 4070 Ti SUPER":GPUSpec(184.6, 0.672, 285),
    "RTX 4070 Ti":      GPUSpec(184.6, 0.504, 285),
    "RTX 4070 SUPER":   GPUSpec(140.7, 0.504, 220),
    "RTX 4070":         GPUSpec(122.4, 0.504, 200),
    "RTX 4060 Ti":      GPUSpec(88.5,  0.288, 160),
    "RTX 4060":         GPUSpec(73.7,  0.272, 115),
    # Consumer RTX 30 series
    "RTX 3090 Ti":      GPUSpec(160.0, 1.008, 450),
    "RTX 3090":         GPUSpec(142.0, 0.936, 350),
    "RTX 3080 Ti":      GPUSpec(136.0, 0.912, 350),
    "RTX 3080":         GPUSpec(119.0, 0.760, 320),
    "RTX 3070 Ti":      GPUSpec(87.0,  0.608, 290),
    "RTX 3070":         GPUSpec(81.0,  0.512, 220),
    "RTX 3060 Ti":      GPUSpec(65.0,  0.448, 200),
    "RTX 3060":         GPUSpec(51.0,  0.360, 170),
    "RTX 3050":         GPUSpec(32.7,  0.224, 130),
    # Consumer RTX 20 series
    "RTX 2080 Ti":      GPUSpec(53.8,  0.616, 250),
    "RTX 2080 SUPER":   GPUSpec(45.2,  0.496, 250),
    "RTX 2080":         GPUSpec(40.3,  0.448, 215),
    "RTX 2070 SUPER":   GPUSpec(36.4,  0.448, 215),
    "RTX 2070":         GPUSpec(29.2,  0.384, 175),
    "RTX 2060 SUPER":   GPUSpec(28.6,  0.384, 175),
    "RTX 2060":         GPUSpec(22.2,  0.336, 160),
    # Laptop
    "RTX 4090 Laptop":  GPUSpec(228.0, 0.576, 150),
    "RTX 4080 Laptop":  GPUSpec(164.0, 0.432, 150),
    "RTX 4070 Laptop":  GPUSpec(117.0, 0.384, 115),
    "RTX 4060 Laptop":  GPUSpec(88.5,  0.256, 115),
    "RTX 3080 Laptop":  GPUSpec(93.0,  0.448, 150),
    "RTX 3070 Laptop":  GPUSpec(64.0,  0.384, 130),
    "RTX 3060 Laptop":  GPUSpec(48.0,  0.336, 115),
    "RTX 3050 Laptop":  GPUSpec(24.0,  0.192, 80),
}


@dataclass
class CrusoeGPUEstimate:
    """Estimation results for a single Crusoe GPU option."""
    name: str
    vram_gb: int
    speedup: float
    est_time_s: float
    est_time_lower_s: float
    est_time_upper_s: float
    on_demand_cost: float
    on_demand_rate: float        # $/hr
    on_demand_cost_lower: float
    on_demand_cost_upper: float
    est_co2_kg: float
    est_energy_kwh: float
    cloud_power_w: float         # estimated power draw
    benchmark_key: str
    scale_factor: float
    scale_note: str
    value_score: float           # cost / hour_saved (lower = better)
    is_best_value: bool = False
    is_fastest: bool = False


def _load_benchmarks() -> dict:
    with open(_DATA_DIR / "gpu_benchmarks.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_crusoe_pricing() -> dict:
    with open(_DATA_DIR / "crusoe_pricing.json", "r", encoding="utf-8") as f:
        return json.load(f)


def detect_gpu_benchmark_key(gpu_name: str) -> Optional[str]:
    """Match an NVML GPU name to a benchmark key (Lambda Labs table only)."""
    if not gpu_name:
        return None

    name = gpu_name.upper().replace("NVIDIA", "").replace("GEFORCE", "").strip()
    sorted_keys = sorted(_GPU_NAME_MAPPING.keys(), key=len, reverse=True)
    for pattern in sorted_keys:
        if pattern.upper() in name:
            match = _GPU_NAME_MAPPING[pattern]
            logger.info(f"GPU '{gpu_name}' matched to benchmark '{match}'")
            return match

    logger.info(f"GPU '{gpu_name}' not in benchmark table, will use specs fallback")
    return None


def detect_gpu_spec(gpu_name: str) -> Optional[Tuple[str, GPUSpec]]:
    """
    Match a GPU name to a spec entry (TFLOPS, bandwidth, TDP).
    Covers far more GPUs than the benchmark table.

    Returns (spec_key, GPUSpec) or None.
    """
    if not gpu_name:
        return None

    name = gpu_name.upper().replace("NVIDIA", "").replace("GEFORCE", "").strip()

    # Try longest keys first for specificity
    sorted_keys = sorted(_GPU_SPECS.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key.upper() in name:
            logger.info(f"GPU '{gpu_name}' matched to spec '{key}'")
            return key, _GPU_SPECS[key]

    logger.warning(f"GPU '{gpu_name}' not found in spec database")
    return None


def get_speedup_from_specs(
    local_spec: GPUSpec,
    crusoe_spec: GPUSpec,
) -> float:
    """
    Estimate speedup ratio using TFLOPS and memory bandwidth.

    Formula (from vice): 70% compute weight + 30% bandwidth weight.
    speedup = crusoe_perf / local_perf
    where perf = 0.7 * tflops + 0.3 * bandwidth (normalized)
    """
    if local_spec.fp16_tflops <= 0:
        return 1.0

    compute_ratio = crusoe_spec.fp16_tflops / local_spec.fp16_tflops

    if (local_spec.memory_bandwidth_tbps > 0
            and crusoe_spec.memory_bandwidth_tbps > 0):
        bw_ratio = crusoe_spec.memory_bandwidth_tbps / local_spec.memory_bandwidth_tbps
        speedup = 0.7 * compute_ratio + 0.3 * bw_ratio
    else:
        speedup = compute_ratio

    return max(speedup, 0.01)  # never negative/zero


def get_speedup_ratio(
    local_gpu_key: str,
    crusoe_gpu_key: str,
    precision: str = "fp16",
    benchmark_task: Optional[str] = None,
    scale_factor: float = 1.0,
) -> Optional[float]:
    """Compute speedup ratio of Crusoe GPU vs local GPU."""
    benchmarks = _load_benchmarks()
    prec_data = benchmarks.get(precision, {})

    local_data = prec_data.get(local_gpu_key)
    crusoe_data = prec_data.get(crusoe_gpu_key)

    if local_data is None or crusoe_data is None:
        return None

    if benchmark_task:
        local_tp = local_data.get(benchmark_task, 0)
        crusoe_tp = crusoe_data.get(benchmark_task, 0)
        if local_tp <= 0 or crusoe_tp <= 0:
            return None
        return (crusoe_tp / local_tp) * scale_factor

    # Geometric mean across all tasks
    log_sum = 0.0
    count = 0
    for task in BENCHMARK_TASKS:
        local_tp = local_data.get(task, 0)
        crusoe_tp = crusoe_data.get(task, 0)
        if local_tp > 0 and crusoe_tp > 0:
            log_sum += math.log(crusoe_tp / local_tp)
            count += 1

    if count == 0:
        return None

    return math.exp(log_sum / count) * scale_factor


def _estimate_cloud_power(
    benchmark_key: str,
    local_efficiency: Optional[float],
) -> float:
    """
    Estimate cloud GPU power draw in watts.

    Strategy: use the GPU's known TDP × estimated efficiency.
    Cloud GPUs in data centers typically run at 75-85% of TDP.
    If we know the local GPU's efficiency (power/TDP ratio), we use
    that as a reference — cloud GPUs tend to run ~5% more efficiently
    due to better cooling.
    """
    tdp = _CLOUD_GPU_TDP.get(benchmark_key, 300)

    if local_efficiency and 0 < local_efficiency <= 1.0:
        # Cloud GPUs run slightly more efficiently than consumer GPUs
        cloud_efficiency = min(local_efficiency * 0.95, 0.90)
    else:
        # Default: 80% of TDP for a typical cloud workload
        cloud_efficiency = 0.80

    return tdp * cloud_efficiency


def estimate_crusoe_options(
    local_time_s: float,
    local_time_lower_s: float,
    local_time_upper_s: float,
    local_gpu_key: Optional[str],
    local_gpu_name: str = "",
    local_gpu_efficiency: Optional[float] = None,
    precision: str = "fp16",
    benchmark_task: Optional[str] = None,
    custom_speedup: Optional[Dict[str, float]] = None,
) -> List[CrusoeGPUEstimate]:
    """
    Estimate training time and cost for all Crusoe GPU options.

    Two estimation strategies:
    1. If local_gpu_key is in the Lambda Labs benchmark table → use
       throughput ratios (most accurate).
    2. Otherwise, fall back to TFLOPS + bandwidth weighted ratio
       (70% compute, 30% memory bandwidth).

    Best value = cost / hours_saved (lowest is best).
    """
    pricing = _load_crusoe_pricing()
    results: List[CrusoeGPUEstimate] = []

    local_hours = local_time_s / 3600.0

    # Resolve local GPU spec for fallback path
    local_spec: Optional[GPUSpec] = None
    if local_gpu_name:
        spec_match = detect_gpu_spec(local_gpu_name)
        if spec_match:
            _, local_spec = spec_match
    # Also try the benchmark key itself as a spec lookup
    if local_spec is None and local_gpu_key:
        local_spec = _GPU_SPECS.get(local_gpu_key)

    for gpu_name, gpu_info in pricing.get("gpus", {}).items():
        benchmark_key = gpu_info["benchmark_key"]
        scale = gpu_info.get("scale_factor", 1.0)
        scale_note = gpu_info.get("scale_note", "")

        # Determine speedup
        speedup = None
        estimation_method = ""

        if custom_speedup and gpu_name in custom_speedup:
            speedup = custom_speedup[gpu_name]
            estimation_method = "custom"
        elif local_gpu_key:
            # Strategy 1: benchmark-based speedup
            speedup = get_speedup_ratio(
                local_gpu_key, benchmark_key, precision,
                benchmark_task, scale,
            )
            if speedup is not None:
                estimation_method = "benchmark"

        if speedup is None and local_spec is not None:
            # Strategy 2: TFLOPS + bandwidth ratio
            crusoe_spec = _GPU_SPECS.get(benchmark_key)
            if crusoe_spec:
                speedup = get_speedup_from_specs(local_spec, crusoe_spec) * scale
                estimation_method = "tflops+bandwidth"
                scale_note = f"Estimated from TFLOPS/bandwidth ratio ({estimation_method})"

        if speedup is None or speedup <= 0:
            continue

        # Estimated times
        est_time = local_time_s / speedup
        est_time_lower = local_time_lower_s / speedup
        est_time_upper = local_time_upper_s / speedup

        # On-demand cost
        on_demand_rate = gpu_info["on_demand_per_gpu_hr"]
        est_hours = est_time / 3600.0
        est_hours_lower = est_time_lower / 3600.0
        est_hours_upper = est_time_upper / 3600.0

        on_demand_cost = est_hours * on_demand_rate
        on_demand_cost_lower = est_hours_lower * on_demand_rate
        on_demand_cost_upper = est_hours_upper * on_demand_rate

        # Cloud power & energy (Crusoe uses 100% renewable → CO2 ≈ 0)
        cloud_power = _estimate_cloud_power(
            benchmark_key, local_gpu_efficiency
        )
        est_energy_kwh = (cloud_power * est_hours) / 1000.0
        est_co2_kg = 0.0  # Crusoe: 100% renewable

        # Value score = cost / hours_saved
        hours_saved = local_hours - est_hours
        if hours_saved > 0:
            value_score = on_demand_cost / hours_saved
        else:
            value_score = float("inf")

        results.append(CrusoeGPUEstimate(
            name=gpu_name,
            vram_gb=gpu_info["vram_gb"],
            speedup=round(speedup, 2),
            est_time_s=est_time,
            est_time_lower_s=est_time_lower,
            est_time_upper_s=est_time_upper,
            on_demand_cost=on_demand_cost,
            on_demand_rate=on_demand_rate,
            on_demand_cost_lower=on_demand_cost_lower,
            on_demand_cost_upper=on_demand_cost_upper,
            est_co2_kg=est_co2_kg,
            est_energy_kwh=est_energy_kwh,
            cloud_power_w=cloud_power,
            benchmark_key=benchmark_key,
            scale_factor=scale,
            scale_note=scale_note,
            value_score=value_score,
        ))

    # Sort by on-demand cost
    results.sort(key=lambda x: x.on_demand_cost)

    if results:
        # Best value = lowest value_score (cost per hour saved)
        best = min(results, key=lambda x: x.value_score)
        best.is_best_value = True
        # Fastest = shortest time
        fastest = min(results, key=lambda x: x.est_time_s)
        fastest.is_fastest = True

    return results
