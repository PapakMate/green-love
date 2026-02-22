"""Tests for co2_equivalences module."""

from green_love.co2_equivalences import (
    compute_equivalences,
    format_co2,
    format_time,
    format_cost,
)


class TestCO2Equivalences:
    def test_compute_equivalences_positive(self):
        equivs = compute_equivalences(1.0)  # 1 kg CO2
        assert len(equivs) == 5

        # Check km driven: 1 kg / 0.246 ≈ 4.07 km
        km = next(e for e in equivs if e.label == "Kilometers driven")
        assert 3.5 < km.value < 4.5

        # Check smartphone charges: 1 kg / 0.00822 ≈ 122
        charges = next(e for e in equivs if e.label == "Smartphone charges")
        assert 100 < charges.value < 150

    def test_compute_equivalences_zero(self):
        equivs = compute_equivalences(0.0)
        assert equivs == []

    def test_compute_equivalences_negative(self):
        equivs = compute_equivalences(-1.0)
        assert equivs == []

    def test_compute_equivalences_small(self):
        equivs = compute_equivalences(0.001)  # 1g CO2
        assert len(equivs) == 5
        # All values should be positive
        for eq in equivs:
            assert eq.value >= 0

    def test_format_co2(self):
        assert "mg" in format_co2(0.0005)
        assert "g" in format_co2(0.5)
        assert "kg" in format_co2(5.0)
        assert "tonnes" in format_co2(1500.0)

    def test_format_time(self):
        assert format_time(30) == "30.0s"
        assert "m" in format_time(120)
        assert "h" in format_time(7200)
        assert "d" in format_time(100000)

    def test_format_cost(self):
        assert "$" in format_cost(0.005)
        assert "$" in format_cost(0.50)
        assert "$" in format_cost(5.00)
        assert "$" in format_cost(500.00)
