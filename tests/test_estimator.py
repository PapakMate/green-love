"""Tests for the GreenLoveEstimator class."""

import time
from unittest.mock import patch, MagicMock

from green_love.estimator import GreenLoveEstimator


class TestEstimatorInit:
    def test_basic_init(self):
        est = GreenLoveEstimator(total_epochs=100)
        assert est.total_epochs == 100
        assert est.is_benchmarking is True

    def test_benchmark_epochs_calculation(self):
        est = GreenLoveEstimator(
            total_epochs=100,
            sample_epochs_pct=10.0,
            warmup_epochs=2,
        )
        # 10% of 100 = 10, but must be >= warmup + 1 = 3
        assert est.benchmark_epochs == 10

    def test_benchmark_epochs_min_warmup(self):
        est = GreenLoveEstimator(
            total_epochs=100,
            sample_epochs_pct=1.0,  # 1% = 1 epoch
            warmup_epochs=3,
        )
        # Must be at least warmup + 1 = 4
        assert est.benchmark_epochs == 4

    def test_benchmark_epochs_cap_at_total(self):
        est = GreenLoveEstimator(
            total_epochs=5,
            sample_epochs_pct=100.0,
            warmup_epochs=2,
        )
        assert est.benchmark_epochs <= 5

    def test_invalid_total_epochs(self):
        try:
            GreenLoveEstimator(total_epochs=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_invalid_precision(self):
        try:
            GreenLoveEstimator(total_epochs=10, precision="bf16")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_invalid_benchmark_task(self):
        try:
            GreenLoveEstimator(total_epochs=10, benchmark_task="invalid_task")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_valid_benchmark_task(self):
        est = GreenLoveEstimator(total_epochs=10, benchmark_task="resnet50")
        assert est.benchmark_task == "resnet50"


class TestEstimatorCallbacks:
    @patch("green_love.estimator.PowerMonitor")
    def test_epoch_timing(self, MockPower):
        """Test that epoch times are recorded."""
        mock_power = MagicMock()
        mock_power.get_gpu_name.return_value = "NVIDIA GeForce RTX 4090"
        mock_power.get_gpu_tdp.return_value = 350.0
        mock_power.get_mean_power_w.return_value = 280.0
        mock_power.get_efficiency.return_value = 0.8
        mock_power.get_samples.return_value = [280.0, 285.0]
        MockPower.return_value = mock_power

        est = GreenLoveEstimator(
            total_epochs=100,
            sample_epochs_pct=5.0,
            warmup_epochs=1,
        )

        # Simulate a warmup epoch
        est.on_epoch_start(0)
        time.sleep(0.01)
        result = est.on_epoch_end(0)
        assert result is True  # should continue
        assert len(est._epoch_times) == 1

    @patch("green_love.estimator.PowerMonitor")
    def test_is_benchmarking_flag(self, MockPower):
        MockPower.return_value = MagicMock()
        est = GreenLoveEstimator(total_epochs=100, warmup_epochs=1, sample_epochs_pct=3.0)
        assert est.is_benchmarking is True
