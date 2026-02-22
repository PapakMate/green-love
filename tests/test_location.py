"""Tests for location module."""

from unittest.mock import patch, MagicMock

from green_love.location import (
    detect_country,
    get_carbon_intensity,
    get_electricity_price,
    get_country_from_timezone,
)


class TestDetectCountry:
    def test_manual_override(self):
        assert detect_country("DE") == "DE"
        assert detect_country("de") == "DE"
        assert detect_country("  us  ") == "US"

    @patch("green_love.location.get_country_from_ip")
    def test_ip_detection(self, mock_ip):
        mock_ip.return_value = "FR"
        assert detect_country() == "FR"

    @patch("green_love.location.get_country_from_ip")
    @patch("green_love.location.get_country_from_timezone")
    def test_timezone_fallback(self, mock_tz, mock_ip):
        mock_ip.return_value = None
        mock_tz.return_value = "JP"
        assert detect_country() == "JP"

    @patch("green_love.location.get_country_from_ip")
    @patch("green_love.location.get_country_from_timezone")
    def test_final_fallback(self, mock_tz, mock_ip):
        mock_ip.return_value = None
        mock_tz.return_value = None
        assert detect_country() == "US"


class TestCarbonIntensity:
    def test_known_country(self):
        intensity, source = get_carbon_intensity("US")
        assert intensity == 388
        assert "US" in source

    def test_manual_override(self):
        intensity, source = get_carbon_intensity("US", manual_intensity=500.0)
        assert intensity == 500.0
        assert "manual" in source

    def test_unknown_country(self):
        intensity, source = get_carbon_intensity("ZZ")
        assert intensity == 494  # world average
        assert "world average" in source

    def test_france_low_carbon(self):
        intensity, _ = get_carbon_intensity("FR")
        assert intensity == 42

    def test_case_insensitive(self):
        i1, _ = get_carbon_intensity("us")
        i2, _ = get_carbon_intensity("US")
        assert i1 == i2


class TestElectricityPrice:
    def test_known_country(self):
        price, source = get_electricity_price("US")
        assert price == 0.184
        assert "US" in source

    def test_manual_override(self):
        price, source = get_electricity_price("US", manual_price=0.30)
        assert price == 0.30
        assert "manual" in source

    def test_unknown_country(self):
        price, source = get_electricity_price("ZZ")
        assert price == 0.167  # world average
        assert "world average" in source

    def test_germany_expensive(self):
        price, _ = get_electricity_price("DE")
        assert price > 0.30  # Germany is expensive
