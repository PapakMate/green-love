"""
CO2 equivalence conversions.

Converts a CO2 amount (in kg) into human-readable real-world equivalences.
All conversion factors sourced from EPA, IEA, and peer-reviewed literature.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class CO2Equivalence:
    """A single CO2 equivalence metric."""
    icon: str
    label: str
    value: float
    unit: str
    description: str


# Conversion factors
_KG_CO2_PER_KM_DRIVEN = 0.246           # avg passenger car (EPA)
_KG_CO2_PER_SMARTPHONE_CHARGE = 0.00822  # EPA
_KG_CO2_PER_LED_HOUR = 0.00388          # 10W LED at US avg grid
_KG_CO2_PER_TREE_DAY = 21.77 / 365.0   # medium growth tree (EPA)
_KG_CO2_PER_GOOGLE_SEARCH = 0.0002      # Google 2009 estimate
_KG_CO2_PARIS_NYC_FLIGHT = 1000.0       # economy round trip per passenger


def compute_equivalences(co2_kg: float) -> List[CO2Equivalence]:
    """
    Convert CO2 emissions (in kg) into real-world equivalence metrics.

    Args:
        co2_kg: Total CO2 emissions in kilograms.

    Returns:
        List of CO2Equivalence objects with human-readable comparisons.
    """
    if co2_kg <= 0:
        return []

    equivalences = [
        CO2Equivalence(
            icon="🚗",
            label="Kilometers driven",
            value=round(co2_kg / _KG_CO2_PER_KM_DRIVEN, 1),
            unit="km",
            description="by an average passenger car",
        ),
        CO2Equivalence(
            icon="📱",
            label="Smartphone charges",
            value=round(co2_kg / _KG_CO2_PER_SMARTPHONE_CHARGE, 0),
            unit="charges",
            description="fully charging a smartphone",
        ),
        CO2Equivalence(
            icon="💡",
            label="LED bulb hours",
            value=round(co2_kg / _KG_CO2_PER_LED_HOUR, 0),
            unit="hours",
            description="running a 10W LED bulb",
        ),
        CO2Equivalence(
            icon="🌳",
            label="Tree-days absorbed",
            value=round(co2_kg / _KG_CO2_PER_TREE_DAY, 1),
            unit="tree-days",
            description="of CO₂ absorption by one tree",
        ),
        CO2Equivalence(
            icon="🔍",
            label="Google searches",
            value=round(co2_kg / _KG_CO2_PER_GOOGLE_SEARCH, 0),
            unit="searches",
            description="on Google",
        ),
        # CO2Equivalence(
        #     icon="✈️",
        #     label="Paris–NYC flights",
        #     value=round(co2_kg / _KG_CO2_PARIS_NYC_FLIGHT * 100, 2),
        #     unit="% of a flight",
        #     description="economy round-trip per passenger",
        # ),
    ]

    return equivalences


def format_co2(co2_kg: float) -> str:
    """Format CO2 in appropriate units (g, kg, or tonnes)."""
    if co2_kg < 0.001:
        return f"{co2_kg * 1_000_000:.1f} mg"
    elif co2_kg < 1.0:
        return f"{co2_kg * 1000:.1f} g"
    elif co2_kg < 1000:
        return f"{co2_kg:.2f} kg"
    else:
        return f"{co2_kg / 1000:.2f} tonnes"


def format_time(seconds: float) -> str:
    """Format duration in human-readable form."""
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


def format_cost(usd: float) -> str:
    """Format USD cost."""
    if usd < 0.01:
        return f"${usd:.4f}"
    elif usd < 1.0:
        return f"${usd:.3f}"
    elif usd < 100:
        return f"${usd:.2f}"
    else:
        return f"${usd:,.2f}"
