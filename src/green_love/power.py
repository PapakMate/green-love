"""
GPU power measurement via NVIDIA Management Library (NVML / pynvml).
Polls GPU power draw in a background thread and provides energy statistics.
"""

import threading
import time
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PowerMonitor:
    """
    Monitors GPU power consumption using pynvml.

    If NVML is unavailable (no NVIDIA GPU or driver), falls back to
    TDP-based estimation using manual_tdp_watts and manual_gpu_utilization.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        poll_interval_s: float = 1.0,
        manual_tdp_watts: Optional[float] = None,
        manual_gpu_utilization: float = 0.70,
    ):
        self.gpu_index = gpu_index
        self.poll_interval_s = poll_interval_s
        self.manual_tdp_watts = manual_tdp_watts
        self.manual_gpu_utilization = manual_gpu_utilization

        self._nvml_available = False
        self._handle = None
        self._samples: List[float] = []  # watts
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._stop_time: Optional[float] = None
        self._gpu_name: Optional[str] = None
        self._gpu_tdp: Optional[float] = manual_tdp_watts

        self._init_nvml()

    def _init_nvml(self):
        """Try to initialize NVML. Gracefully degrade if unavailable."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            self._gpu_name = name

            # Get TDP (power management limit)
            try:
                self._gpu_tdp = pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0
            except Exception:
                try:
                    self._gpu_tdp = pynvml.nvmlDeviceGetEnforcedPowerLimit(self._handle) / 1000.0
                except Exception:
                    self._gpu_tdp = self.manual_tdp_watts

            self._nvml_available = True
            logger.info(f"NVML initialized: {self._gpu_name} (TDP: {self._gpu_tdp}W)")
        except Exception as e:
            logger.warning(f"NVML not available ({e}). Using TDP-based estimation.")
            self._nvml_available = False
            if self.manual_tdp_watts:
                self._gpu_tdp = self.manual_tdp_watts

    def get_gpu_name(self) -> Optional[str]:
        """Return detected GPU name, or None if unavailable."""
        return self._gpu_name

    def get_gpu_tdp(self) -> Optional[float]:
        """Return GPU TDP in watts."""
        return self._gpu_tdp

    def start(self):
        """Start background power sampling."""
        if self._running:
            return
        self._samples = []
        self._running = True
        self._start_time = time.time()
        self._stop_time = None

        if self._nvml_available:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("Power monitoring started (NVML polling)")
        else:
            logger.info("Power monitoring started (TDP estimation mode)")

    def stop(self):
        """Stop background power sampling."""
        self._running = False
        self._stop_time = time.time()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info(f"Power monitoring stopped. {len(self._samples)} samples collected.")

    def _poll_loop(self):
        """Background thread: poll GPU power at regular intervals."""
        import pynvml
        while self._running:
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                power_w = power_mw / 1000.0
                self._samples.append(power_w)
            except Exception as e:
                logger.debug(f"Power poll failed: {e}")
            time.sleep(self.poll_interval_s)

    def get_samples(self) -> List[float]:
        """Return all power samples in watts."""
        return list(self._samples)

    def get_mean_power_w(self) -> float:
        """Return mean power draw in watts."""
        if self._nvml_available and self._samples:
            return sum(self._samples) / len(self._samples)
        elif self._gpu_tdp is not None:
            return self._gpu_tdp * self.manual_gpu_utilization
        return 0.0

    def get_duration_s(self) -> float:
        """Return total monitoring duration in seconds."""
        if self._start_time is None:
            return 0.0
        end = self._stop_time if self._stop_time else time.time()
        return end - self._start_time

    def get_total_energy_kwh(self, duration_s: Optional[float] = None) -> float:
        """
        Return total energy consumed in kWh.

        Args:
            duration_s: Override duration. If None, uses actual monitoring duration.
        """
        if duration_s is None:
            duration_s = self.get_duration_s()
        mean_power_w = self.get_mean_power_w()
        duration_h = duration_s / 3600.0
        return mean_power_w * duration_h / 1000.0  # W -> kW

    def get_efficiency(self) -> Optional[float]:
        """Return GPU power efficiency as fraction (mean_power / TDP)."""
        if self._gpu_tdp and self._gpu_tdp > 0:
            return self.get_mean_power_w() / self._gpu_tdp
        return None

    def cleanup(self):
        """Shutdown NVML."""
        if self._running:
            self.stop()
        if self._nvml_available:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
