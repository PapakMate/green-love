"""
HTML report generator for Green Love.

Builds a Jinja2 context dictionary from BenchmarkResults and renders
the vice-style HTML report template.
"""

import os
import webbrowser
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_VERSION = "0.1.0"


def _fmt_time(seconds: float) -> str:
    """Format duration concisely."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    elif seconds < 86400:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"
    else:
        d = int(seconds // 86400)
        h = int((seconds % 86400) // 3600)
        return f"{d}d {h}h"


def _fmt_cost(usd: float) -> str:
    """Format USD cost."""
    if usd < 0.01:
        return f"${usd:.4f}"
    elif usd < 1.0:
        return f"${usd:.3f}"
    elif usd < 100:
        return f"${usd:.2f}"
    else:
        return f"${usd:,.2f}"


def _fmt_co2(co2_kg: float) -> str:
    """Format CO2 in appropriate unit."""
    if co2_kg < 0.001:
        return f"{co2_kg * 1_000_000:.1f} mg CO₂"
    elif co2_kg < 1.0:
        return f"{co2_kg * 1000:.1f} g CO₂"
    elif co2_kg < 1000:
        return f"{co2_kg:.2f} kg CO₂"
    else:
        return f"{co2_kg / 1000:.2f} tonnes CO₂"


def _fmt_value(v: float) -> str:
    """Format a value nicely (auto-decimal)."""
    if v == float("inf"):
        return "N/A"
    if abs(v) < 0.01:
        return f"{v:.4f}"
    if abs(v) < 10:
        return f"{v:.2f}"
    return f"{v:,.1f}"


def _build_epoch_bars(results) -> list:
    """
    Build epoch timing bar data for the template.
    Includes warmup epochs + measured epochs.
    """
    all_times = results.warmup_times + results.epoch_times
    max_time = max(all_times) if all_times else 1.0

    bars = []
    for i, t in enumerate(results.warmup_times):
        bars.append({
            "label": f"Warmup {i + 1}",
            "time_fmt": f"{t:.2f}s",
            "width_pct": round(t / max_time * 100, 1),
            "bar_class": "bar-warmup",
        })

    for i, t in enumerate(results.epoch_times):
        bars.append({
            "label": f"Epoch {i + 1}",
            "time_fmt": f"{t:.2f}s",
            "width_pct": round(t / max_time * 100, 1),
            "bar_class": "bar-epoch",
        })

    return bars


def _build_crusoe_rows(results) -> list:
    """
    Build Crusoe GPU comparison rows for the template.
    Computes bar widths relative to local time / cost.
    """
    r = results
    local_time = r.est_total_time_s
    local_cost = r.est_total_cost_usd

    # For bar scaling: use local time as 100%
    max_time = local_time if local_time > 0 else 1.0
    # For cost bars: find maximum cost (local electricity vs cloud)
    all_costs = [local_cost] + [e.on_demand_cost for e in r.crusoe_estimates]
    max_cost = max(all_costs) if all_costs else 1.0

    rows = []
    for e in r.crusoe_estimates:
        time_pct = round(e.est_time_s / max_time * 100, 1)
        time_lower_pct = round(e.est_time_lower_s / max_time * 100, 1)
        time_range_pct = round(
            (e.est_time_upper_s - e.est_time_lower_s) / max_time * 100, 1
        )
        cost_pct = round(e.on_demand_cost / max_cost * 100, 1) if max_cost > 0 else 0

        rows.append({
            "name": e.name,
            "vram_gb": e.vram_gb,
            "speedup": f"{e.speedup:.1f}",
            "is_best_value": e.is_best_value,
            "is_fastest": e.is_fastest,
            # Time
            "time_fmt": _fmt_time(e.est_time_s),
            "time_lower_fmt": _fmt_time(e.est_time_lower_s),
            "time_upper_fmt": _fmt_time(e.est_time_upper_s),
            "time_pct": min(time_pct, 100),
            "time_lower_pct": max(time_lower_pct, 0),
            "time_range_pct": max(time_range_pct, 0),
            # Cost
            "cost_fmt": _fmt_cost(e.on_demand_cost),
            "cost_lower_fmt": _fmt_cost(e.on_demand_cost_lower),
            "cost_upper_fmt": _fmt_cost(e.on_demand_cost_upper),
            "cost_pct": min(cost_pct, 100),
            "rate_fmt": _fmt_cost(e.on_demand_rate),
            # CO2
            "est_co2": _fmt_co2(e.est_co2_kg),
            # Energy
            "est_energy_kwh": f"{e.est_energy_kwh:.2f}",
            "cloud_power_w": f"{e.cloud_power_w:.0f}",
            # Value
            "value_score": _fmt_value(e.value_score),
        })

    return rows


def _build_co2_equivalences(results) -> list:
    """Build CO2 equivalence data for template."""
    items = []
    for eq in results.co2_equivalences:
        # Format value nicely
        if eq.value >= 1000:
            vf = f"{eq.value:,.0f}"
        elif eq.value >= 10:
            vf = f"{eq.value:.0f}"
        elif eq.value >= 1:
            vf = f"{eq.value:.1f}"
        else:
            vf = f"{eq.value:.2f}"

        items.append({
            "icon": eq.icon,
            "label": eq.label,
            "value_fmt": vf,
            "unit": eq.unit,
            "description": eq.description,
        })
    return items


def build_report_context(results) -> Dict[str, Any]:
    """
    Build the full Jinja2 context dictionary from BenchmarkResults.
    """
    r = results

    # Best value and fastest GPUs
    best_value = next(
        (e for e in r.crusoe_estimates if e.is_best_value), None
    )
    fastest = next(
        (e for e in r.crusoe_estimates if e.is_fastest), None
    )

    # Carbon source
    carbon_source_live = "live" in r.carbon_intensity_source.lower()

    # GPU efficiency percentage
    gpu_efficiency_pct = round(r.gpu_efficiency * 100) if r.gpu_efficiency else 0

    # Confidence score: weighted from CV and efficiency
    cv_pct = r.cv_epoch_time * 100
    if cv_pct < 3:
        cv_confidence = 100
    elif cv_pct < 10:
        cv_confidence = 70
    else:
        cv_confidence = 40
    eff_contribution = min(gpu_efficiency_pct, 100)
    confidence_score = round(0.6 * cv_confidence + 0.4 * eff_contribution)
    confidence_score = max(0, min(100, confidence_score))

    if confidence_score >= 70:
        confidence_label = "High"
        confidence_class = "confidence-high"
        confidence_color = "var(--accent-green)"
    elif confidence_score >= 40:
        confidence_label = "Medium"
        confidence_class = "confidence-medium"
        confidence_color = "var(--accent-orange)"
    else:
        confidence_label = "Low"
        confidence_class = "confidence-low"
        confidence_color = "var(--accent-red)"

    # Efficiency bar color
    if gpu_efficiency_pct >= 60:
        efficiency_color = "var(--accent-green)"
    elif gpu_efficiency_pct >= 30:
        efficiency_color = "var(--accent-orange)"
    else:
        efficiency_color = "var(--accent-red)"

    # Epoch variance bar (CV scaled, cap at 100%)
    epoch_variance_bar_pct = min(100, round(cv_pct * 5))
    if cv_pct < 5:
        variance_color = "var(--accent-green)"
    elif cv_pct < 15:
        variance_color = "var(--accent-orange)"
    else:
        variance_color = "var(--accent-red)"

    # Warnings
    warnings = []
    if gpu_efficiency_pct < 20:
        warnings.append(
            f"GPU utilization is very low (~{gpu_efficiency_pct}%). "
            "Power readings may not reflect actual training load."
        )
    if cv_pct > 15:
        warnings.append(
            f"Epoch timing variance is high (CV={cv_pct:.1f}%). "
            "Consider running more benchmark epochs for better accuracy."
        )
    if r.sample_data_pct < 5:
        warnings.append(
            f"Only {r.sample_data_pct:.1f}% of data sampled. "
            "Estimates may be less accurate with very small samples."
        )

    # Savings overview (vs best value Crusoe option)
    time_saved = ""
    cost_diff_fmt = ""
    cost_diff_label = ""
    cost_diff_color = "var(--accent-green)"
    if best_value:
        time_saved_s = r.est_total_time_s - best_value.est_time_s
        time_saved = _fmt_time(max(0, time_saved_s))
        money_vs_elec = r.est_total_cost_usd - best_value.on_demand_cost
        if money_vs_elec > 0:
            cost_diff_fmt = f"-{_fmt_cost(abs(money_vs_elec))}"
            cost_diff_label = "Cheaper than local electricity"
            cost_diff_color = "var(--accent-green)"
        else:
            cost_diff_fmt = f"+{_fmt_cost(abs(money_vs_elec))}"
            cost_diff_label = "Cloud premium vs electricity"
            cost_diff_color = "var(--accent-orange)"

    ctx = {
        "version": _VERSION,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        # Local hardware
        "gpu_name": r.gpu_name,
        "precision": r.precision,
        "mean_power_w": f"{r.mean_power_w:.0f}",
        "est_total_time": _fmt_time(r.est_total_time_s),
        "est_total_time_lower": _fmt_time(r.est_total_time_lower_s),
        "est_total_time_upper": _fmt_time(r.est_total_time_upper_s),

        # Benchmark config
        "total_epochs": r.total_epochs,
        "benchmark_epochs": r.benchmark_epochs - r.warmup_epochs,
        "warmup_epochs": r.warmup_epochs,
        "sample_data_pct": f"{r.sample_data_pct:.0f}",
        "sample_data_pct_val": r.sample_data_pct,
        "country_code": r.country_code,
        "benchmark_task": r.benchmark_task,

        # Epoch timing
        "epoch_bars": _build_epoch_bars(r),
        "median_epoch_time": f"{r.median_epoch_time:.2f}",
        "std_epoch_time": f"{r.std_epoch_time:.3f}",
        "cv_epoch_time": f"{r.cv_epoch_time * 100:.1f}",
        "variance_rating": r.variance_rating,
        "est_full_epoch_time": _fmt_time(r.est_full_epoch_time),

        # Cost & CO2
        "est_total_cost": _fmt_cost(r.est_total_cost_usd),
        "est_total_co2": _fmt_co2(r.est_total_co2_kg),
        "electricity_price_kwh": f"${r.electricity_price_kwh:.3f}",
        "est_total_energy_kwh": f"{r.est_total_energy_kwh:.3f}",
        "est_total_hours": f"{r.est_total_time_s / 3600:.1f}",

        # Carbon data
        "carbon_intensity": f"{r.carbon_intensity_gco2_kwh:.0f}",
        "carbon_intensity_source": r.carbon_intensity_source,
        "carbon_source_live": carbon_source_live,
        "electricity_price_source": r.electricity_price_source,

        # Crusoe comparison
        "crusoe_rows": _build_crusoe_rows(r),

        # CO2 equivalences
        "co2_equivalences": _build_co2_equivalences(r),

        # Confidence
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "confidence_class": confidence_class,
        "confidence_color": confidence_color,
        "gpu_efficiency_pct": gpu_efficiency_pct,
        "efficiency_color": efficiency_color,
        "epoch_variance_bar_pct": epoch_variance_bar_pct,
        "variance_color": variance_color,
        "warnings": warnings,

        # Savings overview
        "time_saved": time_saved,
        "cost_diff_fmt": cost_diff_fmt,
        "cost_diff_label": cost_diff_label,
        "cost_diff_color": cost_diff_color,

        # Recommendation
        "best_value": None,
        "fastest": None,
    }

    if best_value:
        ctx["best_value"] = {
            "name": best_value.name,
            "speedup": f"{best_value.speedup:.1f}",
        }
        ctx["best_value_time"] = _fmt_time(best_value.est_time_s)
        ctx["best_value_cost"] = _fmt_cost(best_value.on_demand_cost)
        ctx["best_value_time_lower"] = _fmt_time(best_value.est_time_lower_s)
        ctx["best_value_time_upper"] = _fmt_time(best_value.est_time_upper_s)
        ctx["best_value_cost_lower"] = _fmt_cost(
            best_value.on_demand_cost_lower
        )
        ctx["best_value_cost_upper"] = _fmt_cost(
            best_value.on_demand_cost_upper
        )

    if fastest:
        ctx["fastest"] = {
            "name": fastest.name,
            "speedup": f"{fastest.speedup:.1f}",
        }
        ctx["fastest_time"] = _fmt_time(fastest.est_time_s)
        ctx["fastest_cost"] = _fmt_cost(fastest.on_demand_cost)
        ctx["fastest_time_lower"] = _fmt_time(fastest.est_time_lower_s)
        ctx["fastest_time_upper"] = _fmt_time(fastest.est_time_upper_s)
        ctx["fastest_cost_lower"] = _fmt_cost(fastest.on_demand_cost_lower)
        ctx["fastest_cost_upper"] = _fmt_cost(fastest.on_demand_cost_upper)

    return ctx


def generate_report(
    results,
    output_dir: str = "./crusoe_reports",
    auto_open: bool = True,
) -> str:
    """
    Generate an HTML report from BenchmarkResults.

    Args:
        results: BenchmarkResults dataclass.
        output_dir: Directory to save the report.
        auto_open: Whether to open the report in a browser.

    Returns:
        Path to the generated HTML file.
    """
    # Set up Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=False,
    )
    template = env.get_template("report.html")

    # Build context
    ctx = build_report_context(results)

    # Render
    html = template.render(**ctx)

    # Write file
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"green_love_report_{timestamp}.html"
    filepath = out_path / filename

    filepath.write_text(html, encoding="utf-8")
    logger.info(f"Report saved to {filepath}")
    print(f"\n  📄 Report saved: {filepath}")

    if auto_open:
        try:
            webbrowser.open(str(filepath.resolve()))
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")

    return str(filepath)
