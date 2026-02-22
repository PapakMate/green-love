"""Tests for power module."""

from unittest.mock import patch, MagicMock

from green_love.power import PowerMonitor


class TestPowerMonitorFallback:
    """Test PowerMonitor when NVML is not available."""

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_manual_tdp(self, mock_init):
        """Test TDP-based estimation mode."""
        mon = PowerMonitor(manual_tdp_watts=300.0, manual_gpu_utilization=0.70)
        # In fallback mode, mean power = TDP * utilization
        assert abs(mon.get_mean_power_w() - 210.0) < 0.1

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_energy_calculation(self, mock_init):
        """Test energy calculation from manual TDP."""
        mon = PowerMonitor(manual_tdp_watts=300.0, manual_gpu_utilization=0.70)
        # 210W for 1 hour = 0.21 kWh
        energy = mon.get_total_energy_kwh(duration_s=3600)
        assert abs(energy - 0.21) < 0.01

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_no_tdp(self, mock_init):
        """Test graceful handling when no power info available."""
        mon = PowerMonitor()
        assert mon.get_mean_power_w() == 0.0
        assert mon.get_total_energy_kwh(3600) == 0.0

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_gpu_name_none(self, mock_init):
        """Test GPU name when NVML unavailable."""
        mon = PowerMonitor()
        assert mon.get_gpu_name() is None

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_efficiency_with_tdp(self, mock_init):
        """Test efficiency calculation."""
        mon = PowerMonitor(manual_tdp_watts=300.0, manual_gpu_utilization=0.70)
        mon._gpu_tdp = 300.0
        eff = mon.get_efficiency()
        assert eff is not None
        assert abs(eff - 0.70) < 0.01

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_duration_not_started(self, mock_init):
        """Test duration before start."""
        mon = PowerMonitor()
        assert mon.get_duration_s() == 0.0


class TestPowerMonitorSamples:
    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_samples_empty_before_start(self, mock_init):
        mon = PowerMonitor()
        assert mon.get_samples() == []

    @patch("green_love.power.PowerMonitor._init_nvml")
    def test_manual_samples(self, mock_init):
        """Test with manually injected samples."""
        mon = PowerMonitor(manual_tdp_watts=300.0)
        mon._nvml_available = True
        mon._samples = [200.0, 210.0, 220.0, 215.0]
        mean = mon.get_mean_power_w()
        assert abs(mean - 211.25) < 0.01
