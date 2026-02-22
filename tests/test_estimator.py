"""Tests for the GreenLoveEstimator class (multi-sample benchmark API)."""

import math
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from green_love.estimator import (
    GreenLoveEstimator,
    _spearman_rho,
    _default_train_step,
    _make_subset_loader,
)


# ── Helpers ───────────────────────────────────────────────────────────

def _tiny_model():
    return nn.Linear(4, 2)


def _tiny_dataset(n=200):
    return TensorDataset(
        torch.randn(n, 4),
        torch.randint(0, 2, (n,)),
    )


def _make_estimator(**overrides):
    """Create a GreenLoveEstimator with sensible test defaults."""
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    dataset = _tiny_dataset(200)
    defaults = dict(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataset=dataset,
        total_epochs=10,
        batch_size=32,
        device="cpu",
        manual_tdp_watts=100.0,
    )
    defaults.update(overrides)
    return GreenLoveEstimator(**defaults)


# ── Init validation ──────────────────────────────────────────────────

class TestEstimatorInit:

    def test_basic_init(self):
        est = _make_estimator(total_epochs=50)
        assert est.total_epochs == 50
        assert est.results is None
        assert est.user_chose_continue is False

    def test_invalid_total_epochs(self):
        try:
            _make_estimator(total_epochs=0)
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_invalid_precision(self):
        try:
            _make_estimator(precision="bf16")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_invalid_benchmark_task(self):
        try:
            _make_estimator(benchmark_task="invalid_task")
            assert False, "Should raise ValueError"
        except ValueError:
            pass

    def test_valid_benchmark_task(self):
        est = _make_estimator(benchmark_task="resnet50")
        assert est.benchmark_task == "resnet50"

    def test_default_device_cpu(self):
        est = _make_estimator(device="cpu")
        assert str(est.device) == "cpu"

    def test_warmup_epochs(self):
        est = _make_estimator(warmup_epochs=5)
        assert est.warmup_epochs == 5

    def test_exploration_epochs(self):
        est = _make_estimator(exploration_epochs=20)
        assert est.exploration_epochs == 20

    def test_dataset_size_stored(self):
        ds = _tiny_dataset(500)
        est = _make_estimator(train_dataset=ds)
        assert est._N == 500


# ── Spearman rank correlation ────────────────────────────────────────

class TestSpearman:

    def test_perfect_positive(self):
        assert _spearman_rho([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0

    def test_perfect_negative(self):
        assert _spearman_rho([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0

    def test_near_perfect(self):
        rho = _spearman_rho([1, 2, 3, 4, 5], [10, 21, 29, 42, 50])
        assert rho > 0.9

    def test_small_list(self):
        # With fewer than 3 items, returns 1.0
        assert _spearman_rho([1, 2], [3, 4]) == 1.0


# ── Default train step ───────────────────────────────────────────────

class TestDefaultTrainStep:

    def test_runs_without_error(self):
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(8, 4)
        y = torch.randint(0, 2, (8,))
        _default_train_step(model, (x, y), optimizer, criterion, torch.device("cpu"))

    def test_changes_parameters(self):
        model = _tiny_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()
        w_before = model.weight.data.clone()
        x = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        _default_train_step(model, (x, y), optimizer, criterion, torch.device("cpu"))
        assert not torch.allclose(model.weight.data, w_before)


# ── Subset loader ────────────────────────────────────────────────────

class TestMakeSubsetLoader:

    def test_correct_size(self):
        ds = _tiny_dataset(100)
        loader = _make_subset_loader(ds, n_samples=30, batch_size=8)
        assert len(loader.dataset) == 30

    def test_caps_at_dataset_size(self):
        ds = _tiny_dataset(50)
        loader = _make_subset_loader(ds, n_samples=999, batch_size=8)
        assert len(loader.dataset) == 50

    def test_min_one_sample(self):
        ds = _tiny_dataset(10)
        loader = _make_subset_loader(ds, n_samples=0, batch_size=8)
        assert len(loader.dataset) == 1


# ── Full estimate integration ────────────────────────────────────────

class TestEstimate:

    @patch("green_love.estimator.PowerMonitor")
    @patch("green_love.estimator.generate_report")
    @patch("green_love.estimator.detect_country")
    @patch("green_love.estimator.get_carbon_intensity")
    @patch("green_love.estimator.get_electricity_price")
    @patch("builtins.input", return_value="n")
    def test_estimate_returns_results(
        self, mock_input, mock_ep, mock_ci, mock_country,
        mock_report, MockPower
    ):
        # Setup mocks
        mock_power = MagicMock()
        mock_power.get_gpu_name.return_value = "Test GPU"
        mock_power.get_gpu_tdp.return_value = 100.0
        mock_power.get_mean_power_w.return_value = 80.0
        mock_power.get_efficiency.return_value = 0.8
        mock_power.get_samples.return_value = [80.0]
        MockPower.return_value = mock_power

        mock_country.return_value = "US"
        mock_ci.return_value = (388.0, "test")
        mock_ep.return_value = (0.15, "test")
        mock_report.return_value = "/tmp/test_report.html"

        est = _make_estimator(
            total_epochs=10,
            exploration_epochs=5,
            warmup_epochs=2,
            initial_sample_size=50,
            target_benchmark_time=0.5,
            single_epoch_budget=5.0,
        )
        results = est.estimate()

        assert results is not None
        assert results.total_epochs == 10
        assert results.est_total_time_s > 0
        assert results.est_total_time_lower_s <= results.est_total_time_s
        assert results.est_total_time_s <= results.est_total_time_upper_s
        assert results.full_dataset_size == 200
        assert len(results.sample_sizes_used) >= 1
        assert results.spearman_rho is not None
        assert results.warmup_epochs == 2
        assert est.user_chose_continue is False

    @patch("green_love.estimator.PowerMonitor")
    @patch("green_love.estimator.generate_report")
    @patch("green_love.estimator.detect_country")
    @patch("green_love.estimator.get_carbon_intensity")
    @patch("green_love.estimator.get_electricity_price")
    @patch("builtins.input", return_value="y")
    def test_user_chose_continue(
        self, mock_input, mock_ep, mock_ci, mock_country,
        mock_report, MockPower
    ):
        mock_power = MagicMock()
        mock_power.get_gpu_name.return_value = "Test GPU"
        mock_power.get_gpu_tdp.return_value = 100.0
        mock_power.get_mean_power_w.return_value = 80.0
        mock_power.get_efficiency.return_value = 0.8
        mock_power.get_samples.return_value = [80.0]
        MockPower.return_value = mock_power
        mock_country.return_value = "US"
        mock_ci.return_value = (388.0, "test")
        mock_ep.return_value = (0.15, "test")
        mock_report.return_value = "/tmp/test_report.html"

        est = _make_estimator(
            total_epochs=10,
            exploration_epochs=5,
            warmup_epochs=2,
            initial_sample_size=50,
            target_benchmark_time=0.5,
            single_epoch_budget=5.0,
        )
        est.estimate()
        assert est.user_chose_continue is True

    @patch("green_love.estimator.PowerMonitor")
    @patch("green_love.estimator.generate_report")
    @patch("green_love.estimator.detect_country")
    @patch("green_love.estimator.get_carbon_intensity")
    @patch("green_love.estimator.get_electricity_price")
    @patch("builtins.input", return_value="n")
    def test_model_state_restored(
        self, mock_input, mock_ep, mock_ci, mock_country,
        mock_report, MockPower
    ):
        """Model weights are restored after estimate()."""
        mock_power = MagicMock()
        mock_power.get_gpu_name.return_value = "Test GPU"
        mock_power.get_gpu_tdp.return_value = 100.0
        mock_power.get_mean_power_w.return_value = 80.0
        mock_power.get_efficiency.return_value = 0.8
        mock_power.get_samples.return_value = [80.0]
        MockPower.return_value = mock_power
        mock_country.return_value = "US"
        mock_ci.return_value = (388.0, "test")
        mock_ep.return_value = (0.15, "test")
        mock_report.return_value = "/tmp/test_report.html"

        model = _tiny_model()
        w_before = model.weight.data.clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        dataset = _tiny_dataset(200)

        est = GreenLoveEstimator(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataset=dataset,
            total_epochs=10,
            batch_size=32,
            device="cpu",
            exploration_epochs=5,
            warmup_epochs=2,
            initial_sample_size=50,
            target_benchmark_time=0.5,
            single_epoch_budget=5.0,
            manual_tdp_watts=100.0,
        )
        est.estimate()

        # Weights should be restored to original
        assert torch.allclose(model.weight.data, w_before)
