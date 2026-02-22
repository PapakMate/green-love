"""Tests for report module."""

import os
import tempfile
from unittest.mock import patch

from green_love.report import generate_report, build_report_context
from green_love.co2_equivalences import compute_equivalences, CO2Equivalence
from green_love.benchmarks import CrusoeGPUEstimate


def _make_mock_results():
    """Create a mock BenchmarkResults for testing."""

    class MockResults:
        # Multi-sample data
        sample_sizes_used = [500, 750, 1125]
        full_dataset_size = 10000
        spearman_rho = 0.98
        warmup_epoch_estimates = [20.0, 18.0, 16.0]
        steady_epoch_estimate = 15.25
        steady_epoch_std = 0.82

        epoch_times = [1.5, 1.6, 1.4, 1.55]
        warmup_times = [2.0, 1.8]
        median_epoch_time = 1.525
        mean_epoch_time = 1.5125
        std_epoch_time = 0.082
        cv_epoch_time = 0.054
        ci_lower_epoch = 1.39
        ci_upper_epoch = 1.66
        variance_rating = "Good"

        est_full_epoch_time = 15.25
        est_full_epoch_lower = 13.9
        est_full_epoch_upper = 16.6

        est_total_time_s = 15250.0
        est_total_time_lower_s = 13900.0
        est_total_time_upper_s = 16600.0

        mean_power_w = 280.0
        gpu_tdp_w = 350.0
        gpu_efficiency = 0.80
        power_samples = [275.0, 280.0, 285.0, 278.0, 282.0]
        est_total_energy_kwh = 1.186

        carbon_intensity_gco2_kwh = 388.0
        carbon_intensity_source = "offline table (US)"
        electricity_price_kwh = 0.184
        electricity_price_source = "offline table (US)"
        est_total_co2_kg = 0.460
        est_total_cost_usd = 0.218

        country_code = "US"
        gpu_name = "NVIDIA GeForce RTX 4090"
        gpu_benchmark_key = "RTX 4090"

        crusoe_estimates = [
            CrusoeGPUEstimate(
                name="A40 48GB",
                vram_gb=48,
                speedup=0.85,
                est_time_s=17941.0,
                est_time_lower_s=16353.0,
                est_time_upper_s=19529.0,
                on_demand_cost=4.49,
                on_demand_rate=0.90,
                on_demand_cost_lower=4.09,
                on_demand_cost_upper=4.88,
                est_co2_kg=0.0,
                est_energy_kwh=1.5,
                cloud_power_w=270.0,
                benchmark_key="RTX A6000",
                scale_factor=0.9,
                scale_note="test",
                value_score=float("inf"),
                is_best_value=True,
                is_fastest=False,
            ),
            CrusoeGPUEstimate(
                name="H100 HGX 80GB",
                vram_gb=80,
                speedup=2.15,
                est_time_s=7093.0,
                est_time_lower_s=6465.0,
                est_time_upper_s=7721.0,
                on_demand_cost=7.68,
                on_demand_rate=3.90,
                on_demand_cost_lower=7.00,
                on_demand_cost_upper=8.36,
                est_co2_kg=0.0,
                est_energy_kwh=0.9,
                cloud_power_w=560.0,
                benchmark_key="H100 80GB SXM5",
                scale_factor=1.0,
                scale_note="Direct match",
                value_score=3.39,
                is_best_value=False,
                is_fastest=True,
            ),
        ]

        co2_savings_kg = 0.460
        co2_equivalences = compute_equivalences(0.460)

        total_epochs = 100
        benchmark_epochs = 6
        warmup_epochs = 2
        sample_data_pct = 10.0
        precision = "fp16"
        benchmark_task = None

    return MockResults()


class TestReportGeneration:
    def test_generates_html_file(self):
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(results, output_dir=tmpdir, auto_open=False)
            assert os.path.exists(path)
            assert path.endswith(".html")

    def test_html_contains_key_sections(self):
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(results, output_dir=tmpdir, auto_open=False)
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()

            # Check key sections exist
            assert "Green Love" in html
            assert "RTX 4090" in html
            assert "Benchmark Config" in html
            assert "Epoch Timing" in html
            assert "Local Hardware" in html
            assert "Crusoe Cloud GPU Comparison" in html
            assert "Carbon Savings" in html
            assert "Recommendation" in html

    def test_html_contains_crusoe_gpus(self):
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(results, output_dir=tmpdir, auto_open=False)
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()

            assert "A40 48GB" in html
            assert "H100 HGX 80GB" in html
            assert "Best Value" in html
            assert "Fastest" in html

    def test_html_uses_css_bars_not_chartjs(self):
        """Report uses pure CSS bars, no Chart.js."""
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report(results, output_dir=tmpdir, auto_open=False)
            with open(path, "r", encoding="utf-8") as f:
                html = f.read()

            # No Chart.js dependency
            assert "chart.js" not in html.lower()
            # Uses CSS bar classes instead
            assert "bar-container" in html
            assert "bar-crusoe" in html

    def test_creates_report_dir(self):
        results = _make_mock_results()
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = os.path.join(tmpdir, "nested", "reports")
            path = generate_report(results, output_dir=report_dir, auto_open=False)
            assert os.path.exists(report_dir)
            assert os.path.exists(path)


class TestReportContext:
    def test_context_has_required_keys(self):
        results = _make_mock_results()
        ctx = build_report_context(results)
        required_keys = [
            "version", "timestamp", "gpu_name", "precision",
            "mean_power_w", "est_total_time", "total_epochs",
            "benchmark_epochs", "warmup_epochs", "sample_data_pct",
            "country_code", "epoch_bars", "median_epoch_time",
            "crusoe_rows", "co2_equivalences", "carbon_intensity",
            "carbon_intensity_source", "est_total_cost", "est_total_co2",
            "best_value", "fastest",
        ]
        for key in required_keys:
            assert key in ctx, f"Missing key: {key}"

    def test_context_best_value_and_fastest(self):
        results = _make_mock_results()
        ctx = build_report_context(results)
        assert ctx["best_value"] is not None
        assert ctx["best_value"]["name"] == "A40 48GB"
        assert ctx["fastest"] is not None
        assert ctx["fastest"]["name"] == "H100 HGX 80GB"

    def test_epoch_bars_count(self):
        results = _make_mock_results()
        ctx = build_report_context(results)
        # 2 warmup + 4 measured = 6 bars
        assert len(ctx["epoch_bars"]) == 6

    def test_crusoe_rows_count(self):
        results = _make_mock_results()
        ctx = build_report_context(results)
        assert len(ctx["crusoe_rows"]) == 2
