"""
GreenLoveEstimator — PyTorch training callback (pure callback pattern).

The user keeps full control of their training loop, data loading, etc.
The estimator only needs on_epoch_start / on_epoch_end calls.
It automatically accounts for the data sampling percentage when
projecting full-training estimates.
"""

import math
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .power import PowerMonitor
from .location import detect_country, get_carbon_intensity, get_electricity_price
from .benchmarks import (
    detect_gpu_benchmark_key,
    estimate_crusoe_options,
    CrusoeGPUEstimate,
    BENCHMARK_TASKS,
)
from .co2_equivalences import (
    compute_equivalences,
    format_time,
    format_cost,
    format_co2,
    CO2Equivalence,
)
from .report import generate_report

logger = logging.getLogger(__name__)

# t-distribution critical values for 95% CI (two-tailed, alpha=0.025)
_T_CRITICAL = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042, 40: 2.021, 50: 2.009, 60: 2.000,
    80: 1.990, 100: 1.984,
}


def _get_t_critical(df: int) -> float:
    if df <= 0:
        return 12.706
    if df in _T_CRITICAL:
        return _T_CRITICAL[df]
    keys = sorted(_T_CRITICAL.keys())
    if df > keys[-1]:
        return 1.96
    for i in range(len(keys) - 1):
        if keys[i] <= df < keys[i + 1]:
            t1, t2 = _T_CRITICAL[keys[i]], _T_CRITICAL[keys[i + 1]]
            frac = (df - keys[i]) / (keys[i + 1] - keys[i])
            return t1 + frac * (t2 - t1)
    return 1.96


@dataclass
class BenchmarkResults:
    """Results from the benchmark phase."""
    # Epoch timing (raw benchmark measurements)
    epoch_times: List[float]       # seconds per measured epoch (benchmark data %)
    warmup_times: List[float]      # seconds per warmup epoch
    median_epoch_time: float       # median of measured epoch times (benchmark)
    mean_epoch_time: float
    std_epoch_time: float
    cv_epoch_time: float
    ci_lower_epoch: float
    ci_upper_epoch: float
    variance_rating: str

    # Estimated full-data epoch time (scaled by data %)
    est_full_epoch_time: float
    est_full_epoch_lower: float
    est_full_epoch_upper: float

    # Local full-training estimates
    est_total_time_s: float
    est_total_time_lower_s: float
    est_total_time_upper_s: float

    # Power & energy
    mean_power_w: float
    gpu_tdp_w: Optional[float]
    gpu_efficiency: Optional[float]
    power_samples: List[float]
    est_total_energy_kwh: float

    # CO2 & cost
    carbon_intensity_gco2_kwh: float
    carbon_intensity_source: str
    electricity_price_kwh: float
    electricity_price_source: str
    est_total_co2_kg: float
    est_total_cost_usd: float

    # Location
    country_code: str

    # GPU info
    gpu_name: str
    gpu_benchmark_key: Optional[str]

    # Crusoe comparisons
    crusoe_estimates: List[CrusoeGPUEstimate]

    # Carbon savings on Crusoe (local CO2 - best Crusoe CO2)
    co2_savings_kg: float
    co2_equivalences: List[CO2Equivalence]

    # Config
    total_epochs: int
    benchmark_epochs: int
    warmup_epochs: int
    sample_data_pct: float
    precision: str
    benchmark_task: Optional[str]


class GreenLoveEstimator:
    """
    PyTorch training callback — pure callback pattern.

    The user controls their training loop entirely. The estimator only
    needs ``on_epoch_start(epoch)`` and ``on_epoch_end(epoch)`` calls.

    The ``sample_data_pct`` parameter tells the estimator what fraction
    of the full dataset is being used during benchmark epochs so it can
    scale the measured epoch time up to the full-data estimate.

    Usage::

        estimator = GreenLoveEstimator(total_epochs=100)

        # Phase 1: Benchmark (estimator times first ~10% of epochs)
        for epoch in range(100):
            estimator.on_epoch_start(epoch)
            for batch in loader:
                ...  # training step
            if not estimator.on_epoch_end(epoch):
                break

        # Phase 2: Full training (if user chose to continue locally)
        if estimator.user_chose_continue:
            for epoch in range(100):
                for batch in loader:
                    ...  # training step
    """

    def __init__(
        self,
        total_epochs: int,
        sample_data_pct: float = 100.0,
        sample_epochs_pct: float = 10.0,
        warmup_epochs: int = 2,
        # Location & environment
        country_code: Optional[str] = None,
        carbon_intensity: Optional[float] = None,
        electricity_price: Optional[float] = None,
        electricity_maps_api_key: Optional[str] = None,
        # GPU
        gpu_name: Optional[str] = None,
        gpu_index: int = 0,
        manual_tdp_watts: Optional[float] = None,
        manual_gpu_utilization: float = 0.70,
        # Benchmark comparison
        benchmark_task: Optional[str] = None,
        precision: str = "fp16",
        custom_speedup: Optional[Dict[str, float]] = None,
        # Report
        report_dir: str = "./crusoe_reports",
        auto_open_report: bool = True,
        # Power monitoring
        power_poll_interval: float = 1.0,
    ):
        if total_epochs < 1:
            raise ValueError("total_epochs must be >= 1")
        if not 0 < sample_data_pct <= 100:
            raise ValueError("sample_data_pct must be between 0 and 100")
        if not 0 < sample_epochs_pct <= 100:
            raise ValueError("sample_epochs_pct must be between 0 and 100")
        if precision not in ("fp16", "fp32"):
            raise ValueError("precision must be 'fp16' or 'fp32'")
        if benchmark_task and benchmark_task not in BENCHMARK_TASKS:
            raise ValueError(
                f"benchmark_task must be one of {BENCHMARK_TASKS}, "
                f"got '{benchmark_task}'"
            )

        self.total_epochs = total_epochs
        self.sample_data_pct = sample_data_pct
        self.sample_epochs_pct = sample_epochs_pct
        self.warmup_epochs = warmup_epochs
        self.benchmark_task = benchmark_task
        self.precision = precision
        self.custom_speedup = custom_speedup
        self.report_dir = report_dir
        self.auto_open_report = auto_open_report

        # Compute benchmark epochs
        sample_epochs = max(
            1, math.ceil(total_epochs * sample_epochs_pct / 100.0)
        )
        self.benchmark_epochs = max(warmup_epochs + 1, sample_epochs)
        self.benchmark_epochs = min(self.benchmark_epochs, total_epochs)

        # State
        self._epoch_times: List[float] = []
        self._epoch_start_time: Optional[float] = None
        self._benchmark_done = False
        self._results: Optional[BenchmarkResults] = None
        self._user_chose_continue = False

        # Overrides
        self._country_code_override = country_code
        self._carbon_intensity_override = carbon_intensity
        self._electricity_price_override = electricity_price
        self._electricity_maps_api_key = electricity_maps_api_key
        self._gpu_name_override = gpu_name

        # Power monitor
        self._power = PowerMonitor(
            gpu_index=gpu_index,
            poll_interval_s=power_poll_interval,
            manual_tdp_watts=manual_tdp_watts,
            manual_gpu_utilization=manual_gpu_utilization,
        )

        logger.info("=" * 60)
        logger.info("Green Love Estimator initialized")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Benchmark epochs: {self.benchmark_epochs}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Sample data: {sample_data_pct}%")
        logger.info(f"  Precision: {precision}")
        if benchmark_task:
            logger.info(f"  Benchmark task: {benchmark_task}")
        logger.info("=" * 60)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_benchmarking(self) -> bool:
        """True while still in the benchmark phase."""
        return not self._benchmark_done

    @property
    def results(self) -> Optional[BenchmarkResults]:
        """Benchmark results (available after benchmark completes)."""
        return self._results

    @property
    def user_chose_continue(self) -> bool:
        """True if the user chose to continue training locally."""
        return self._user_chose_continue

    # ── Callbacks ─────────────────────────────────────────────────────────

    def on_epoch_start(self, epoch: int) -> None:
        """Call at the start of each training epoch (0-indexed)."""
        self._epoch_start_time = time.time()

        if epoch == 0:
            self._power.start()
            print("\n" + "=" * 60)
            print("  🔬 Green Love — Benchmark Phase")
            print(f"  Running {self.benchmark_epochs} benchmark epochs "
                  f"({self.warmup_epochs} warmup + "
                  f"{self.benchmark_epochs - self.warmup_epochs} measured)")
            if self.sample_data_pct < 100:
                print(f"  Using {self.sample_data_pct}% of data "
                      f"(estimates scaled to 100%)")
            print("=" * 60 + "\n")

        if epoch < self.warmup_epochs:
            print(f"  ⏳ Epoch {epoch + 1}/{self.benchmark_epochs} (warmup)")
        elif epoch < self.benchmark_epochs:
            print(f"  📊 Epoch {epoch + 1}/{self.benchmark_epochs} (measuring)")

    def on_epoch_end(self, epoch: int) -> bool:
        """
        Call at the end of each training epoch.
        Returns True → continue, False → stop.
        """
        if self._epoch_start_time is None:
            return True

        elapsed = time.time() - self._epoch_start_time
        self._epoch_times.append(elapsed)

        if epoch == self.benchmark_epochs - 1 and not self._benchmark_done:
            self._benchmark_done = True
            self._power.stop()
            return self._process_benchmark()

        return True

    # ── Internal ──────────────────────────────────────────────────────────

    def _process_benchmark(self) -> bool:
        print("\n" + "=" * 60)
        print("  📈 Processing benchmark results...")
        print("=" * 60)

        warmup_times = self._epoch_times[:self.warmup_epochs]
        measured_times = self._epoch_times[self.warmup_epochs:self.benchmark_epochs]

        if not measured_times:
            print("  ⚠️  No measured epochs. Cannot estimate.")
            return True

        median_time = statistics.median(measured_times)
        mean_time = statistics.mean(measured_times)
        n = len(measured_times)

        if n >= 2:
            std_time = statistics.stdev(measured_times)
            cv = std_time / mean_time if mean_time > 0 else 0
            t_crit = _get_t_critical(n - 1)
            margin = t_crit * std_time / math.sqrt(n)
            ci_lower = max(0, median_time - margin)
            ci_upper = median_time + margin
        else:
            std_time = 0.0
            cv = 0.0
            ci_lower = median_time * 0.9
            ci_upper = median_time * 1.1

        if cv < 0.05:
            variance_rating = "Excellent"
        elif cv < 0.15:
            variance_rating = "Good"
        elif cv < 0.30:
            variance_rating = "Fair"
        else:
            variance_rating = "Poor"

        # --- Scale from benchmark data% to full data ---
        data_scale = 100.0 / self.sample_data_pct
        est_full_epoch = median_time * data_scale
        est_full_epoch_lower = ci_lower * data_scale
        est_full_epoch_upper = ci_upper * data_scale

        # --- Total local estimate (full-data epoch × total epochs) ---
        est_total_time = est_full_epoch * self.total_epochs
        est_total_time_lower = est_full_epoch_lower * self.total_epochs
        est_total_time_upper = est_full_epoch_upper * self.total_epochs

        # --- Power & energy ---
        mean_power = self._power.get_mean_power_w()
        gpu_tdp = self._power.get_gpu_tdp()
        gpu_efficiency = self._power.get_efficiency()
        power_samples = self._power.get_samples()

        est_total_hours = est_total_time / 3600.0
        est_total_energy = mean_power * est_total_hours / 1000.0  # kWh

        # --- Location & carbon/price ---
        country = detect_country(self._country_code_override)
        ci_val, ci_source = self._get_carbon_intensity(country)
        ep_val, ep_source = get_electricity_price(
            country, self._electricity_price_override
        )

        est_co2_kg = est_total_energy * ci_val / 1000.0
        est_cost = est_total_energy * ep_val

        # --- GPU identification ---
        gpu_name = (self._gpu_name_override
                    or self._power.get_gpu_name()
                    or "Unknown GPU")
        gpu_key = detect_gpu_benchmark_key(gpu_name)

        # --- Crusoe comparison ---
        # Works with benchmark key (accurate) or falls back to specs
        crusoe_estimates = estimate_crusoe_options(
            local_time_s=est_total_time,
            local_time_lower_s=est_total_time_lower,
            local_time_upper_s=est_total_time_upper,
            local_gpu_key=gpu_key,
            local_gpu_name=gpu_name,
            local_gpu_efficiency=gpu_efficiency,
            precision=self.precision,
            benchmark_task=self.benchmark_task,
            custom_speedup=self.custom_speedup,
        )

        # --- Carbon savings ---
        best_crusoe_co2 = min(
            (e.est_co2_kg for e in crusoe_estimates), default=0
        )
        co2_savings = est_co2_kg - best_crusoe_co2
        co2_equivs = compute_equivalences(co2_savings)

        # --- Build results ---
        self._results = BenchmarkResults(
            epoch_times=measured_times,
            warmup_times=warmup_times,
            median_epoch_time=median_time,
            mean_epoch_time=mean_time,
            std_epoch_time=std_time,
            cv_epoch_time=cv,
            ci_lower_epoch=ci_lower,
            ci_upper_epoch=ci_upper,
            variance_rating=variance_rating,
            est_full_epoch_time=est_full_epoch,
            est_full_epoch_lower=est_full_epoch_lower,
            est_full_epoch_upper=est_full_epoch_upper,
            est_total_time_s=est_total_time,
            est_total_time_lower_s=est_total_time_lower,
            est_total_time_upper_s=est_total_time_upper,
            mean_power_w=mean_power,
            gpu_tdp_w=gpu_tdp,
            gpu_efficiency=gpu_efficiency,
            power_samples=power_samples,
            est_total_energy_kwh=est_total_energy,
            carbon_intensity_gco2_kwh=ci_val,
            carbon_intensity_source=ci_source,
            electricity_price_kwh=ep_val,
            electricity_price_source=ep_source,
            est_total_co2_kg=est_co2_kg,
            est_total_cost_usd=est_cost,
            country_code=country,
            gpu_name=gpu_name,
            gpu_benchmark_key=gpu_key,
            crusoe_estimates=crusoe_estimates,
            co2_savings_kg=co2_savings,
            co2_equivalences=co2_equivs,
            total_epochs=self.total_epochs,
            benchmark_epochs=self.benchmark_epochs,
            warmup_epochs=self.warmup_epochs,
            sample_data_pct=self.sample_data_pct,
            precision=self.precision,
            benchmark_task=self.benchmark_task,
        )

        report_path = generate_report(
            self._results, self.report_dir, self.auto_open_report
        )
        self._print_summary(report_path)
        self._user_chose_continue = self._prompt_continue()
        # Always stop the current loop — user restarts fresh if continuing
        return False

    def _get_carbon_intensity(self, country: str):
        """Try Electricity Maps API if key given, else offline table."""
        if self._carbon_intensity_override is not None:
            return self._carbon_intensity_override, "manual override"

        api_key = self._electricity_maps_api_key
        if not api_key:
            import os
            api_key = os.environ.get("ELECTRICITY_MAPS_API_KEY", "")

        if api_key:
            try:
                import requests
                zone = country.upper()
                url = (f"https://api.electricitymap.org/v3/"
                       f"carbon-intensity/latest?zone={zone}")
                resp = requests.get(
                    url, headers={"auth-token": api_key}, timeout=5
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "carbonIntensity" in data:
                        val = data["carbonIntensity"]
                        logger.info(
                            f"Electricity Maps: {val} gCO2/kWh "
                            f"for zone {zone}"
                        )
                        return val, f"Electricity Maps live ({zone})"
            except Exception as e:
                logger.warning(f"Electricity Maps API failed: {e}")

        return get_carbon_intensity(country, None)

    def _print_summary(self, report_path: str) -> None:
        r = self._results
        if r is None:
            return

        best_value = next(
            (e for e in r.crusoe_estimates if e.is_best_value), None
        )
        fastest = next(
            (e for e in r.crusoe_estimates if e.is_fastest), None
        )

        print()
        print("╔" + "═" * 62 + "╗")
        print("║  🔬 GREEN LOVE — BENCHMARK COMPLETE"
              + " " * 10 + "║")
        print("╠" + "═" * 62 + "╣")
        print(f"║  📄 Report: {report_path:<49}║")
        print("╠" + "═" * 62 + "╣")
        print(f"║  🖥️  Local GPU: {r.gpu_name:<45}║")
        bm = format_time(r.median_epoch_time)
        fe = format_time(r.est_full_epoch_time)
        print(f"║  ⏱️  Benchmark epoch: {bm}  →  "
              f"Full-data epoch: {fe:<16}║")
        print(f"║  📊 Variance: {r.cv_epoch_time:.1%} CV "
              f"({r.variance_rating})"
              f"{' ' * (36 - len(r.variance_rating))}║")
        print("╠" + "═" * 62 + "╣")
        line = (f"║  ⏱️  Est. total time: "
                f"{format_time(r.est_total_time_s)}")
        ci = (f" ({format_time(r.est_total_time_lower_s)}–"
              f"{format_time(r.est_total_time_upper_s)})")
        print(f"{(line + ci):<63}║")
        print(f"║  💰 Est. local cost: "
              f"{format_cost(r.est_total_cost_usd):<40}║")
        print(f"║  🌍 Est. local CO₂: "
              f"{format_co2(r.est_total_co2_kg):<40}║")

        if best_value:
            print("╠" + "═" * 62 + "╣")
            line = f"║  🏆 Best value: {best_value.name}"
            print(f"{line:<63}║")
            line = (
                f"║     {format_time(best_value.est_time_s)}, "
                f"{format_cost(best_value.on_demand_cost)}, "
                f"{best_value.speedup:.1f}x faster"
            )
            print(f"{line:<63}║")

        if fastest and fastest != best_value:
            line = f"║  ⚡ Fastest: {fastest.name}"
            print(f"{line:<63}║")
            line = (
                f"║     {format_time(fastest.est_time_s)}, "
                f"{format_cost(fastest.on_demand_cost)}, "
                f"{fastest.speedup:.1f}x faster"
            )
            print(f"{line:<63}║")

        print("╚" + "═" * 62 + "╝")

    def _prompt_continue(self) -> bool:
        print()
        try:
            answer = input(
                "  Continue training locally? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = "n"

        if answer in ("y", "yes"):
            print("\n  ✅ Continuing training...\n")
            return True
        else:
            print("\n  🛑 Training stopped.\n")
            return False

    def cleanup(self):
        """Clean up resources."""
        self._power.cleanup()
