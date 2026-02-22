"""
GreenLoveEstimator — PyTorch training time & cost estimator.

Uses adaptive multi-sample benchmarking to accurately estimate total
training time using linear scaling laws and statistical inference.
Model and optimizer states are saved before benchmarking and restored
after, so the user can continue training from the original state.
"""

import copy
import math
import time
import logging
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

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


# ── Default training step ─────────────────────────────────────────────

def _default_train_step(model, batch, optimizer, criterion, device):
    """Default training step: forward → loss → backward → step."""
    inputs, targets = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()


# ── Spearman rank correlation ────────────────────────────────────────

def _spearman_rho(x: List[float], y: List[float]) -> float:
    """Compute Spearman rank correlation without scipy."""
    n = len(x)
    if n < 3:
        return 1.0

    def _rank(vals):
        indexed = sorted(range(n), key=lambda i: vals[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(indexed):
            ranks[idx] = float(rank_val + 1)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d_sq / (n * (n ** 2 - 1))


# ── Subset helper ────────────────────────────────────────────────────

def _make_subset_loader(
    dataset: Dataset,
    n_samples: int,
    batch_size: int,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader from a random subset of *n_samples* items."""
    total = len(dataset)
    n_samples = max(1, min(n_samples, total))
    gen = torch.Generator()
    gen.manual_seed(seed)
    indices = torch.randperm(total, generator=gen)[:n_samples].tolist()
    subset = Subset(dataset, sorted(indices))
    return DataLoader(
        subset,
        batch_size=min(batch_size, n_samples),
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )


@dataclass
class BenchmarkResults:
    """Results from the multi-sample benchmark phase."""

    # ── Multi-sample data ────────────────────────────────────────────
    sample_sizes_used: List[int]
    full_dataset_size: int
    spearman_rho: float

    # Scaled epoch estimates at full dataset size N
    warmup_epoch_estimates: List[float]   # [e1(N), e2(N), e3(N)]
    steady_epoch_estimate: float          # Ae_bar(N)
    steady_epoch_std: float               # sigma(N)

    # ── Report-compatible fields ─────────────────────────────────────
    epoch_times: List[float]       # raw steady-state scaled epoch times
    warmup_times: List[float]      # raw warmup scaled epoch times
    median_epoch_time: float       # Ae_bar(N) (alias for report)
    mean_epoch_time: float
    std_epoch_time: float
    cv_epoch_time: float
    ci_lower_epoch: float
    ci_upper_epoch: float
    variance_rating: str

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

    # Carbon savings
    co2_savings_kg: float
    co2_equivalences: List[CO2Equivalence]

    # Raw multi-sample data: {sample_size: [epoch_times]}
    sample_epoch_data: Dict[int, List[float]]

    # Config
    total_epochs: int
    benchmark_epochs: int
    warmup_epochs: int
    sample_data_pct: float
    precision: str
    benchmark_task: Optional[str]


class GreenLoveEstimator:
    """
    PyTorch training time & cost estimator.

    Uses adaptive multi-sample benchmarking: finds a representative
    sample size, trains at several sample sizes, and uses cross-sample
    linear scaling to estimate full-training time.

    Usage::

        estimator = GreenLoveEstimator(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_dataset=dataset,
            total_epochs=100,
            batch_size=64,
        )
        results = estimator.estimate()

        if estimator.user_chose_continue:
            # run your full training loop
            ...
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_dataset: Dataset,
        total_epochs: int,
        batch_size: int = 64,
        device: Optional[str] = None,
        train_step: Optional[Callable] = None,
        # Benchmark tuning
        exploration_epochs: int = 30,
        warmup_epochs: int = 3,
        target_benchmark_time: float = 10.0,
        initial_sample_size: int = 1000,
        single_epoch_budget: float = 0.3,
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
        if precision not in ("fp16", "fp32"):
            raise ValueError("precision must be 'fp16' or 'fp32'")
        if benchmark_task and benchmark_task not in BENCHMARK_TASKS:
            raise ValueError(
                f"benchmark_task must be one of {BENCHMARK_TASKS}, "
                f"got '{benchmark_task}'"
            )

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dataset = train_dataset
        self.total_epochs = total_epochs
        self.batch_size = batch_size
        self.train_step = train_step or _default_train_step

        # Device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Benchmark tuning
        self.exploration_epochs = exploration_epochs
        self.warmup_epochs = warmup_epochs
        self.target_benchmark_time = target_benchmark_time
        self.initial_sample_size = initial_sample_size
        self.single_epoch_budget = single_epoch_budget

        # Other config
        self.benchmark_task = benchmark_task
        self.precision = precision
        self.custom_speedup = custom_speedup
        self.report_dir = report_dir
        self.auto_open_report = auto_open_report

        # State
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

        self._N = len(train_dataset)

        logger.info("=" * 60)
        logger.info("Green Love Estimator initialized")
        logger.info(f"  Total epochs: {total_epochs}")
        logger.info(f"  Dataset size: {self._N}")
        logger.info(f"  Exploration epochs: {exploration_epochs}")
        logger.info(f"  Warmup epochs: {warmup_epochs}")
        logger.info(f"  Target benchmark time: {target_benchmark_time}s")
        logger.info(f"  Precision: {precision}")
        if benchmark_task:
            logger.info(f"  Benchmark task: {benchmark_task}")
        logger.info("=" * 60)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def results(self) -> Optional[BenchmarkResults]:
        """Benchmark results (available after estimate() completes)."""
        return self._results

    @property
    def user_chose_continue(self) -> bool:
        """True if the user chose to continue training locally."""
        return self._user_chose_continue

    # ── Main entry point ──────────────────────────────────────────────

    def estimate(self) -> BenchmarkResults:
        """
        Run the full estimation pipeline.

        1. Save model/optimizer state
        2. Find representative sample size (adaptive)
        3. Train at multiple sample sizes, collect epoch times
        4. Compute scaled estimates using cross-sample averaging
        5. Build confidence intervals, validate linearity
        6. Restore model/optimizer state
        7. Compute power, CO2, cost, Crusoe comparisons
        8. Generate report and prompt user
        """
        print("\n" + "=" * 60)
        print("  🔬 Green Love — Multi-Sample Benchmark")
        print(f"  Dataset: {self._N} samples")
        print(f"  Target epochs: {self.total_epochs}")
        print("=" * 60 + "\n")

        # Save initial state
        initial_model_state = copy.deepcopy(self.model.state_dict())
        initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # Warm up CUDA before any timed work
        self._warmup_cuda()

        # Start power monitoring
        self._power.start()

        try:
            # Step 1: Find representative sample size
            repr_n, repr_epoch_times = self._find_representative_sample_size()
            print(f"\n  ✅ Representative sample size: {repr_n}\n")

            # Step 2: Multi-sample benchmark (up then down from repr_n)
            print(f"  📊 Running multi-sample benchmark...\n")
            all_data = self._run_multi_sample_benchmark(
                repr_n, repr_epoch_times
            )
            print(f"\n  ✅ Collected data from {len(all_data)} sample sizes\n")

            # Step 4: Process results
            results = self._process_benchmark(all_data)

        finally:
            self._power.stop()
            # Restore initial state
            self.model.load_state_dict(initial_model_state)
            self.optimizer.load_state_dict(initial_optimizer_state)

        self._results = results

        report_path = generate_report(
            results, self.report_dir, self.auto_open_report
        )
        self._print_summary(report_path)
        self._user_chose_continue = self._prompt_continue()

        return results

    # ── Step 1: Find representative sample size ──────────────────────

    def _find_representative_sample_size(self) -> Tuple[int, List[float]]:
        """
        Adaptive algorithm:
        Start with (n, Be). Iteratively adjust n:
          Case 1: first epoch > budget → n = n/10, retry
          Case 2: 1 ≤ completed < Be → n_new = n * completed (EXIT)
          Case 3: all Be epochs done (too fast) → n = n * target/total, retry
        Returns (representative_n, epoch_times_from_last_run).
        """
        n = min(self.initial_sample_size, self._N)
        Be = self.exploration_epochs
        budget = self.single_epoch_budget
        target = self.target_benchmark_time
        max_retries = 15
        min_n = max(1, self.batch_size)

        print(f"  🔍 Finding representative sample size "
              f"(start n={n}, target={target}s)...")

        for attempt in range(max_retries):
            self._reset_model_optimizer()
            loader = _make_subset_loader(
                self.train_dataset, n, self.batch_size, seed=42 + attempt
            )

            epoch_times: List[float] = []
            case1_hit = False

            for epoch in range(Be):
                self._cuda_sync()
                t0 = time.time()
                self._run_one_epoch(loader)
                self._cuda_sync()
                elapsed = time.time() - t0
                epoch_times.append(elapsed)

                if elapsed > budget:
                    completed = len(epoch_times)
                    if completed == 1:
                        # Case 1: first epoch > budget → shrink by 10x
                        n_new = max(n // 10, min_n)
                        print(f"    Attempt {attempt + 1}: "
                              f"epoch={elapsed:.3f}s > {budget}s "
                              f"→ n: {n} → {n_new}")
                        if n_new == n:
                            print(f"    Cannot reduce further (min={min_n})")
                            return n, epoch_times
                        n = n_new
                        case1_hit = True
                        break
                    else:
                        # Case 2: 1 ≤ completed < Be → EXIT
                        n_new = max(min_n, min(n * completed, self._N))
                        print(f"    Attempt {attempt + 1}: "
                              f"{completed}/{Be} epochs "
                              f"→ n_new = {n}*{completed} = {n_new}  (EXIT)")
                        # Re-run at n_new to collect clean epoch times
                        self._reset_model_optimizer()
                        loader2 = _make_subset_loader(
                            self.train_dataset, n_new, self.batch_size
                        )
                        final_times: List[float] = []
                        for _e in range(Be):
                            self._cuda_sync()
                            t0 = time.time()
                            self._run_one_epoch(loader2)
                            self._cuda_sync()
                            final_times.append(time.time() - t0)
                        return n_new, final_times

            if case1_hit:
                continue  # retry with smaller n

            # All Be epochs completed
            total_time = sum(epoch_times)

            # Case 3: all epochs done, scale to target time
            if total_time < 1e-6:
                n_new = min(n * 100, self._N)
            else:
                n_new = int(n * target / total_time)
            n_new = max(min_n, min(n_new, self._N))

            print(f"    Attempt {attempt + 1}: all {Be} epochs in "
                  f"{total_time:.2f}s → n: {n} → {n_new}")

            if n_new == n or abs(n_new - n) / max(n, 1) < 0.1:
                n = n_new
                return n, epoch_times
            n = n_new

        # Fallback: return whatever we have
        self._reset_model_optimizer()
        loader = _make_subset_loader(
            self.train_dataset, n, self.batch_size, seed=99
        )
        epoch_times = []
        for _e in range(Be):
            self._cuda_sync()
            t0 = time.time()
            self._run_one_epoch(loader)
            self._cuda_sync()
            epoch_times.append(time.time() - t0)
        return n, epoch_times

    # ── Step 2: Multi-sample benchmark (up then down) ────────────────

    def _run_multi_sample_benchmark(
        self, repr_n: int, repr_epoch_times: List[float]
    ) -> Dict[int, List[float]]:
        """
        From representative n:
        - Re-train at repr_n for fair baseline (algorithm data is stale)
        - Scale UP: n*1.5, n*1.5^2, ... until total_time > 20s (include last)
        - Scale DOWN: n/1.5, n/1.5^2, ... until total_time < 1s (include last)
        Returns {sample_size: [epoch_times]}.
        """
        Be = self.exploration_epochs
        # Allow sample sizes smaller than batch_size for downscaling
        # (partial batches are fine for timing)
        min_n = max(1, self.batch_size // 8)
        all_data: Dict[int, List[float]] = {}

        # Re-train at repr_n for a fair baseline (algorithm data was
        # collected under different thermal/GPU conditions)
        self._reset_model_optimizer()
        loader = _make_subset_loader(
            self.train_dataset, repr_n, self.batch_size
        )
        base_epoch_times: List[float] = []
        for epoch in range(Be):
            self._cuda_sync()
            t0 = time.time()
            self._run_one_epoch(loader)
            self._cuda_sync()
            base_epoch_times.append(time.time() - t0)

        all_data[repr_n] = base_epoch_times
        base_total = sum(base_epoch_times)
        print(f"    n={repr_n:>8d}  total={base_total:>7.2f}s  "
              f"avg_epoch={base_total / Be:.4f}s  (base)")

        # ── Scale UP: multiply by 1.5 until total > 20s ──────────────
        n_up = repr_n
        for i in range(1, 50):
            n_next = int(round(n_up * 1.5))
            if n_next == n_up or n_next > self._N:
                break
            n_up = n_next

            self._reset_model_optimizer()
            loader = _make_subset_loader(
                self.train_dataset, n_up, self.batch_size
            )
            epoch_times = []
            for epoch in range(Be):
                self._cuda_sync()
                t0 = time.time()
                self._run_one_epoch(loader)
                self._cuda_sync()
                epoch_times.append(time.time() - t0)

            total = sum(epoch_times)
            all_data[n_up] = epoch_times
            print(f"    n={n_up:>8d}  total={total:>7.2f}s  "
                  f"avg_epoch={total / Be:.4f}s  ↑")

            if total > 20.0:
                break  # logged it, now stop going up

        # ── Scale DOWN: divide by 1.5 until total < 1s ──────────────
        n_down = repr_n
        for i in range(1, 50):
            n_next = int(round(n_down / 1.5))
            if n_next == n_down or n_next < min_n:
                break
            n_down = n_next

            self._reset_model_optimizer()
            loader = _make_subset_loader(
                self.train_dataset, n_down, self.batch_size
            )
            epoch_times = []
            for epoch in range(Be):
                self._cuda_sync()
                t0 = time.time()
                self._run_one_epoch(loader)
                self._cuda_sync()
                epoch_times.append(time.time() - t0)

            total = sum(epoch_times)
            all_data[n_down] = epoch_times
            print(f"    n={n_down:>8d}  total={total:>7.2f}s  "
                  f"avg_epoch={total / Be:.4f}s  ↓")

            if total < 1.0:
                break  # logged it, now stop going down

        return all_data

    # ── Step 3: Process benchmark data ───────────────────────────────

    def _process_benchmark(
        self, all_data: Dict[int, List[float]]
    ) -> BenchmarkResults:
        """
        Compute scaled estimates from multi-sample data:
        - Ae_bar(N) = mean(e_{i,k} * N/k) for i > warmup, all k
        - e_j(N) = mean(e_{j,k} * N/k) for j in {1,2,3}
        - sigma(N) = std(e_{i,k} * N/k) for i > warmup
        """
        print("  📈 Computing scaled estimates...")

        N = self._N
        warmup = self.warmup_epochs

        # Collect scaled epoch times
        e_warmup_scaled: List[List[float]] = [[] for _ in range(warmup)]
        steady_scaled: List[float] = []

        for k, times in all_data.items():
            scale = N / k
            # Warmup epochs (e1, e2, e3)
            for j in range(min(warmup, len(times))):
                e_warmup_scaled[j].append(times[j] * scale)
            # Steady-state epochs (i > warmup)
            for t in times[warmup:]:
                steady_scaled.append(t * scale)

        # Warmup estimates e1(N), e2(N), e3(N)
        warmup_estimates = [
            statistics.mean(vals) if vals else 0.0
            for vals in e_warmup_scaled
        ]

        # Steady-state estimate Ae_bar(N)
        if steady_scaled:
            ae_bar = statistics.mean(steady_scaled)
            n_steady = len(steady_scaled)
            if n_steady >= 2:
                sigma = statistics.stdev(steady_scaled)
                cv = sigma / ae_bar if ae_bar > 0 else 0.0
            else:
                sigma = 0.0
                cv = 0.0
        else:
            # Fallback: use all epoch times
            all_scaled = []
            for vals in e_warmup_scaled:
                all_scaled.extend(vals)
            ae_bar = statistics.mean(all_scaled) if all_scaled else 0.001
            sigma = (statistics.stdev(all_scaled)
                     if len(all_scaled) >= 2 else 0.0)
            cv = sigma / ae_bar if ae_bar > 0 else 0.0
            n_steady = len(all_scaled)

        # ── Total time estimate ──────────────────────────────────────
        Be = self.total_epochs
        if Be > warmup:
            est_total = (sum(warmup_estimates)
                         + (Be - warmup) * ae_bar)
        else:
            est_total = sum(warmup_estimates[:Be])

        # ── Confidence interval (normal distribution z=1.96) ─────────
        # We have lots of data points (all e_{i,k} * N/k for all
        # post-warmup epochs i and all sample sizes k), so the normal
        # approximation is sufficient — no t-distribution needed.
        z_alpha = 1.96  # 95% CI, two-tailed
        margin = (math.sqrt(Be) * sigma * z_alpha
                  / math.sqrt(max(n_steady, 1)))

        est_total_lower = max(0, est_total - margin)
        est_total_upper = est_total + margin

        # Epoch-level CI
        epoch_margin = z_alpha * sigma / math.sqrt(max(n_steady, 1))
        ci_lower_epoch = max(0, ae_bar - epoch_margin)
        ci_upper_epoch = ae_bar + epoch_margin

        # Variance rating
        if cv < 0.05:
            variance_rating = "Excellent"
        elif cv < 0.15:
            variance_rating = "Good"
        elif cv < 0.30:
            variance_rating = "Fair"
        else:
            variance_rating = "Poor"

        # ── Spearman validation ──────────────────────────────────────
        sizes_list = sorted(all_data.keys())
        times_list = [sum(all_data[k]) for k in sizes_list]
        rho = _spearman_rho(
            [float(s) for s in sizes_list],
            times_list,
        )
        if rho < 0.9:
            logger.warning(
                f"Spearman ρ = {rho:.3f} — linear scaling assumption "
                "may not hold for this model."
            )

        # ── Power & energy ───────────────────────────────────────────
        mean_power = self._power.get_mean_power_w()
        gpu_tdp = self._power.get_gpu_tdp()
        gpu_efficiency = self._power.get_efficiency()
        power_samples = self._power.get_samples()

        est_total_hours = est_total / 3600.0
        est_total_energy = mean_power * est_total_hours / 1000.0

        # ── Location & carbon/price ──────────────────────────────────
        country = detect_country(self._country_code_override)
        ci_val, ci_source = self._get_carbon_intensity(country)
        ep_val, ep_source = get_electricity_price(
            country, self._electricity_price_override
        )

        est_co2_kg = est_total_energy * ci_val / 1000.0
        est_cost = est_total_energy * ep_val

        # ── GPU identification ───────────────────────────────────────
        gpu_name = (self._gpu_name_override
                    or self._power.get_gpu_name()
                    or "Unknown GPU")
        gpu_key = detect_gpu_benchmark_key(gpu_name)

        # ── Crusoe comparison ────────────────────────────────────────
        crusoe_estimates = estimate_crusoe_options(
            local_time_s=est_total,
            local_time_lower_s=est_total_lower,
            local_time_upper_s=est_total_upper,
            local_gpu_key=gpu_key,
            local_gpu_name=gpu_name,
            local_gpu_efficiency=gpu_efficiency,
            precision=self.precision,
            benchmark_task=self.benchmark_task,
            custom_speedup=self.custom_speedup,
        )

        # ── Carbon savings ───────────────────────────────────────────
        best_crusoe_co2 = min(
            (e.est_co2_kg for e in crusoe_estimates), default=0
        )
        co2_savings = est_co2_kg - best_crusoe_co2
        co2_equivs = compute_equivalences(co2_savings)

        # ── Compute smallest sample pct for report compat ────────────
        min_sample = min(all_data.keys()) if all_data else self._N
        sample_data_pct = min_sample / self._N * 100.0

        # Total benchmark epochs run across all sample sizes
        total_bench_epochs = sum(len(t) for t in all_data.values())

        # Build warmup/epoch times lists for epoch bars in report
        # (use representative warmup and steady-state from all data)
        warmup_times_for_report = warmup_estimates
        steady_times_for_report = steady_scaled[:20]  # cap for display

        # Full-epoch CI
        est_full_epoch_lower = max(0, ae_bar - epoch_margin)
        est_full_epoch_upper = ae_bar + epoch_margin

        return BenchmarkResults(
            sample_sizes_used=sizes_list,
            full_dataset_size=N,
            spearman_rho=rho,
            warmup_epoch_estimates=warmup_estimates,
            steady_epoch_estimate=ae_bar,
            steady_epoch_std=sigma,
            epoch_times=steady_times_for_report,
            warmup_times=warmup_times_for_report,
            median_epoch_time=ae_bar,
            mean_epoch_time=ae_bar,
            std_epoch_time=sigma,
            cv_epoch_time=cv,
            ci_lower_epoch=ci_lower_epoch,
            ci_upper_epoch=ci_upper_epoch,
            variance_rating=variance_rating,
            est_full_epoch_time=ae_bar,
            est_full_epoch_lower=est_full_epoch_lower,
            est_full_epoch_upper=est_full_epoch_upper,
            est_total_time_s=est_total,
            est_total_time_lower_s=est_total_lower,
            est_total_time_upper_s=est_total_upper,
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
            sample_epoch_data=dict(all_data),
            total_epochs=self.total_epochs,
            benchmark_epochs=total_bench_epochs,
            warmup_epochs=self.warmup_epochs,
            sample_data_pct=sample_data_pct,
            precision=self.precision,
            benchmark_task=self.benchmark_task,
        )

    # ── Training helpers ─────────────────────────────────────────────

    def _cuda_sync(self) -> None:
        """Synchronize CUDA if using GPU (makes time.time() accurate)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _warmup_cuda(self) -> None:
        """Run a few throwaway passes to warm up CUDA JIT / cuDNN."""
        if self.device.type != "cuda":
            return
        print("  ⚡ Warming up CUDA...")
        self._reset_model_optimizer()
        loader = _make_subset_loader(
            self.train_dataset,
            min(self.batch_size * 2, self._N),
            self.batch_size,
        )
        for _ in range(3):
            self._run_one_epoch(loader)
        torch.cuda.synchronize(self.device)
        print("  ⚡ CUDA warm-up done.\n")

    def _run_one_epoch(self, loader: DataLoader) -> None:
        """Run a single training epoch using the train_step function."""
        self.model.train()
        for batch in loader:
            self.train_step(
                self.model, batch, self.optimizer,
                self.criterion, self.device,
            )

    def _reset_model_optimizer(self) -> None:
        """Reset model and optimizer to initial random state."""
        # Re-init model weights
        def _init_weights(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.model.apply(_init_weights)
        # Reset optimizer state (momentum buffers, Adam moments, etc.)
        self.optimizer.state.clear()

    # ── Carbon intensity ─────────────────────────────────────────────

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

    # ── Summary printing ─────────────────────────────────────────────

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
        fe = format_time(r.est_full_epoch_time)
        print(f"║  ⏱️  Est. full-data epoch: {fe:<35}║")
        print(f"║  📊 Variance: {r.cv_epoch_time:.1%} CV "
              f"({r.variance_rating})"
              f"{' ' * (36 - len(r.variance_rating))}║")
        rho_str = f"{r.spearman_rho:.3f}"
        print(f"║  📈 Spearman ρ: {rho_str}"
              f"  ({len(r.sample_sizes_used)} sample sizes)"
              f"{' ' * (30 - len(rho_str))}║")
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
