"""Tests for benchmarks module."""

from green_love.benchmarks import (
    detect_gpu_benchmark_key,
    detect_gpu_spec,
    get_speedup_from_specs,
    get_speedup_ratio,
    estimate_crusoe_options,
    GPUSpec,
    BENCHMARK_TASKS,
)


class TestGPUDetection:
    def test_rtx_4090(self):
        assert detect_gpu_benchmark_key("NVIDIA GeForce RTX 4090") == "RTX 4090"

    def test_a100_sxm(self):
        assert detect_gpu_benchmark_key("NVIDIA A100-SXM4-80GB") == "A100 80GB SXM4"

    def test_h100_sxm(self):
        assert detect_gpu_benchmark_key("NVIDIA H100 SXM") == "H100 80GB SXM5"

    def test_v100(self):
        assert detect_gpu_benchmark_key("Tesla V100-SXM2-16GB") == "V100 16GB"

    def test_rtx_3090(self):
        assert detect_gpu_benchmark_key("NVIDIA GeForce RTX 3090") == "RTX 3090"

    def test_unknown_gpu(self):
        assert detect_gpu_benchmark_key("Intel UHD 630") is None

    def test_empty_string(self):
        assert detect_gpu_benchmark_key("") is None

    def test_none(self):
        assert detect_gpu_benchmark_key(None) is None


class TestSpeedupRatio:
    def test_same_gpu(self):
        ratio = get_speedup_ratio("RTX 4090", "RTX 4090", "fp16")
        assert ratio is not None
        assert abs(ratio - 1.0) < 0.01

    def test_faster_gpu_fp16(self):
        ratio = get_speedup_ratio("RTX 4090", "H100 80GB SXM5", "fp16")
        assert ratio is not None
        assert ratio > 1.0  # H100 should be faster

    def test_specific_task(self):
        ratio = get_speedup_ratio("RTX 4090", "H100 80GB SXM5", "fp16", "resnet50")
        assert ratio is not None
        assert ratio > 1.0

    def test_fp32(self):
        ratio = get_speedup_ratio("RTX 4090", "A100 80GB SXM4", "fp32")
        assert ratio is not None
        assert ratio > 0

    def test_scale_factor(self):
        ratio_no_scale = get_speedup_ratio("RTX 4090", "RTX 6000 Ada", "fp16")
        ratio_scaled = get_speedup_ratio("RTX 4090", "RTX 6000 Ada", "fp16", scale_factor=0.85)
        assert ratio_no_scale is not None
        assert ratio_scaled is not None
        assert abs(ratio_scaled - ratio_no_scale * 0.85) < 0.01

    def test_invalid_gpu(self):
        ratio = get_speedup_ratio("RTX 4090", "NonexistentGPU", "fp16")
        assert ratio is None


class TestCrusoeEstimates:
    def test_basic_estimates(self):
        results = estimate_crusoe_options(
            local_time_s=3600,       # 1 hour
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
            precision="fp16",
        )
        assert len(results) > 0

        for gpu in results:
            assert gpu.est_time_s > 0
            assert gpu.on_demand_cost > 0
            assert gpu.speedup > 0
            assert gpu.est_co2_kg >= 0  # Should be ~0 for Crusoe

    def test_best_value_marked(self):
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
        )
        best_values = [g for g in results if g.is_best_value]
        assert len(best_values) == 1

    def test_fastest_marked(self):
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
        )
        fastest = [g for g in results if g.is_fastest]
        assert len(fastest) == 1

    def test_custom_speedup(self):
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
            custom_speedup={"H100 HGX 80GB": 3.0},
        )
        h100 = next((g for g in results if g.name == "H100 HGX 80GB"), None)
        assert h100 is not None
        assert abs(h100.speedup - 3.0) < 0.01
        assert abs(h100.est_time_s - 1200) < 1  # 3600 / 3.0

    def test_sorted_by_cost(self):
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
        )
        costs = [g.on_demand_cost for g in results]
        assert costs == sorted(costs)

    def test_confidence_intervals_propagated(self):
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3000,
            local_time_upper_s=4200,
            local_gpu_key="RTX 4090",
        )
        for gpu in results:
            assert gpu.est_time_lower_s <= gpu.est_time_s <= gpu.est_time_upper_s
            assert gpu.on_demand_cost_lower <= gpu.on_demand_cost <= gpu.on_demand_cost_upper


class TestGPUSpecDetection:
    """Tests for the GPU spec database and detection."""

    def test_detect_rtx_3050(self):
        result = detect_gpu_spec("NVIDIA GeForce RTX 3050")
        assert result is not None
        key, spec = result
        assert "3050" in key
        assert spec.fp16_tflops > 0
        assert spec.memory_bandwidth_tbps > 0

    def test_detect_rtx_4090(self):
        result = detect_gpu_spec("NVIDIA GeForce RTX 4090")
        assert result is not None
        key, spec = result
        assert "4090" in key
        assert spec.fp16_tflops > 100  # RTX 4090 has very high TFLOPS

    def test_detect_h100(self):
        result = detect_gpu_spec("NVIDIA H100 SXM")
        assert result is not None
        key, spec = result
        assert "H100" in key
        assert spec.fp16_tflops > 200

    def test_detect_a100(self):
        result = detect_gpu_spec("NVIDIA A100-SXM4-80GB")
        assert result is not None
        key, spec = result
        assert spec.fp16_tflops > 0

    def test_detect_unknown_gpu(self):
        result = detect_gpu_spec("Intel UHD 630")
        assert result is None

    def test_detect_empty(self):
        result = detect_gpu_spec("")
        assert result is None

    def test_detect_none(self):
        result = detect_gpu_spec(None)
        assert result is None


class TestSpeedupFromSpecs:
    """Tests for the TFLOPS + bandwidth weighted speedup formula."""

    def test_same_gpu_returns_one(self):
        spec = GPUSpec(fp16_tflops=100.0, memory_bandwidth_tbps=2.0, tdp_watts=300)
        speedup = get_speedup_from_specs(spec, spec)
        assert abs(speedup - 1.0) < 0.01

    def test_faster_gpu_returns_above_one(self):
        local = GPUSpec(fp16_tflops=32.7, memory_bandwidth_tbps=0.224, tdp_watts=130)  # ~RTX 3050
        crusoe = GPUSpec(fp16_tflops=989.0, memory_bandwidth_tbps=3.35, tdp_watts=700)  # ~H100 SXM
        speedup = get_speedup_from_specs(local, crusoe)
        assert speedup > 5.0  # H100 should be much faster than RTX 3050

    def test_slower_gpu_returns_below_one(self):
        local = GPUSpec(fp16_tflops=989.0, memory_bandwidth_tbps=3.35, tdp_watts=700)  # H100
        crusoe = GPUSpec(fp16_tflops=32.7, memory_bandwidth_tbps=0.224, tdp_watts=130)  # RTX 3050
        speedup = get_speedup_from_specs(local, crusoe)
        assert 0 < speedup < 1.0

    def test_zero_tflops_returns_one(self):
        local = GPUSpec(fp16_tflops=0, memory_bandwidth_tbps=1.0, tdp_watts=300)
        crusoe = GPUSpec(fp16_tflops=100.0, memory_bandwidth_tbps=2.0, tdp_watts=300)
        speedup = get_speedup_from_specs(local, crusoe)
        assert speedup == 1.0

    def test_formula_weights(self):
        """Verify 70% compute + 30% bandwidth weighting."""
        local = GPUSpec(fp16_tflops=100.0, memory_bandwidth_tbps=1.0, tdp_watts=300)
        crusoe = GPUSpec(fp16_tflops=200.0, memory_bandwidth_tbps=3.0, tdp_watts=350)
        speedup = get_speedup_from_specs(local, crusoe)
        expected = 0.7 * (200 / 100) + 0.3 * (3.0 / 1.0)  # 1.4 + 0.9 = 2.3
        assert abs(speedup - expected) < 0.01

    def test_zero_bandwidth_falls_back_to_compute_only(self):
        local = GPUSpec(fp16_tflops=100.0, memory_bandwidth_tbps=0, tdp_watts=300)
        crusoe = GPUSpec(fp16_tflops=300.0, memory_bandwidth_tbps=0, tdp_watts=350)
        speedup = get_speedup_from_specs(local, crusoe)
        assert abs(speedup - 3.0) < 0.01  # pure compute ratio


class TestSpecBasedEstimates:
    """Tests for Crusoe estimates using the spec-based fallback path."""

    def test_rtx_3050_uses_spec_fallback(self):
        """RTX 3050 is NOT in benchmark table so must use spec fallback."""
        assert detect_gpu_benchmark_key("NVIDIA GeForce RTX 3050") is None

        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key=None,
            local_gpu_name="NVIDIA GeForce RTX 3050",
            precision="fp16",
        )
        assert len(results) > 0
        for gpu in results:
            assert gpu.speedup > 1.0  # All Crusoe GPUs should be faster than RTX 3050
            assert gpu.est_time_s > 0
            assert gpu.on_demand_cost > 0

    def test_no_gpu_key_no_name_returns_empty(self):
        """Without any GPU info, no estimates can be made."""
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key=None,
            local_gpu_name="",
        )
        assert len(results) == 0

    def test_spec_fallback_has_best_value_and_fastest(self):
        """Spec-based results should still have best_value and fastest flags."""
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key=None,
            local_gpu_name="NVIDIA GeForce RTX 3050",
        )
        assert len(results) > 0
        assert any(g.is_best_value for g in results)
        assert any(g.is_fastest for g in results)

    def test_benchmark_path_preferred_over_specs(self):
        """When benchmark data exists, it should be used over spec fallback."""
        results = estimate_crusoe_options(
            local_time_s=3600,
            local_time_lower_s=3200,
            local_time_upper_s=4000,
            local_gpu_key="RTX 4090",
            local_gpu_name="NVIDIA GeForce RTX 4090",
        )
        # RTX 4090 is in benchmark table, so results should exist
        assert len(results) > 0
