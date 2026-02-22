"""
Geolocation and carbon intensity / electricity price lookup.

Cascade for location detection:
  1. User-provided country_code parameter
  2. ip-api.com (free, no key, returns countryCode)
  3. System timezone -> country mapping fallback
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"

# Timezone prefix -> ISO country code mapping (common ones)
_TZ_TO_COUNTRY = {
    "America/New_York": "US", "America/Chicago": "US", "America/Denver": "US",
    "America/Los_Angeles": "US", "America/Phoenix": "US", "America/Anchorage": "US",
    "Pacific/Honolulu": "US", "America/Detroit": "US", "America/Indiana": "US",
    "America/Boise": "US", "America/Juneau": "US", "America/Adak": "US",
    "America/Toronto": "CA", "America/Vancouver": "CA", "America/Edmonton": "CA",
    "America/Winnipeg": "CA", "America/Halifax": "CA", "America/Montreal": "CA",
    "Europe/London": "GB", "Europe/Berlin": "DE", "Europe/Paris": "FR",
    "Europe/Rome": "IT", "Europe/Madrid": "ES", "Europe/Amsterdam": "NL",
    "Europe/Brussels": "BE", "Europe/Zurich": "CH", "Europe/Vienna": "AT",
    "Europe/Stockholm": "SE", "Europe/Oslo": "NO", "Europe/Copenhagen": "DK",
    "Europe/Helsinki": "FI", "Europe/Warsaw": "PL", "Europe/Prague": "CZ",
    "Europe/Budapest": "HU", "Europe/Bucharest": "RO", "Europe/Sofia": "BG",
    "Europe/Athens": "GR", "Europe/Lisbon": "PT", "Europe/Dublin": "IE",
    "Europe/Zagreb": "HR", "Europe/Belgrade": "RS", "Europe/Bratislava": "SK",
    "Europe/Ljubljana": "SI", "Europe/Tallinn": "EE", "Europe/Riga": "LV",
    "Europe/Vilnius": "LT", "Europe/Luxembourg": "LU", "Europe/Valletta": "MT",
    "Europe/Nicosia": "CY", "Europe/Istanbul": "TR", "Europe/Moscow": "RU",
    "Europe/Kiev": "UA", "Europe/Minsk": "BY",
    "Asia/Tokyo": "JP", "Asia/Shanghai": "CN", "Asia/Hong_Kong": "CN",
    "Asia/Seoul": "KR", "Asia/Kolkata": "IN", "Asia/Taipei": "TW",
    "Asia/Singapore": "SG", "Asia/Bangkok": "TH", "Asia/Jakarta": "ID",
    "Asia/Kuala_Lumpur": "MY", "Asia/Manila": "PH", "Asia/Ho_Chi_Minh": "VN",
    "Asia/Dhaka": "BD", "Asia/Karachi": "PK", "Asia/Riyadh": "SA",
    "Asia/Dubai": "AE", "Asia/Tel_Aviv": "IL", "Asia/Jerusalem": "IL",
    "Asia/Almaty": "KZ",
    "Australia/Sydney": "AU", "Australia/Melbourne": "AU",
    "Australia/Brisbane": "AU", "Australia/Perth": "AU",
    "Pacific/Auckland": "NZ",
    "America/Mexico_City": "MX", "America/Sao_Paulo": "BR",
    "America/Buenos_Aires": "AR", "America/Santiago": "CL",
    "America/Bogota": "CO", "America/Lima": "PE",
    "Africa/Cairo": "EG", "Africa/Lagos": "NG",
    "Africa/Johannesburg": "ZA", "Africa/Nairobi": "KE",
    "Atlantic/Reykjavik": "IS",
}


def _load_json(filename: str) -> dict:
    """Load a JSON data file from the data directory."""
    filepath = _DATA_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_country_from_ip() -> Optional[str]:
    """
    Detect country code using ip-api.com (free, no key required).
    Returns ISO 2-letter country code or None on failure.
    """
    try:
        import requests
        resp = requests.get("http://ip-api.com/json/", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == "success":
                cc = data.get("countryCode")
                city = data.get("city", "")
                country = data.get("country", "")
                logger.info(f"Geolocation: {city}, {country} ({cc})")
                return cc
    except Exception as e:
        logger.debug(f"ip-api.com lookup failed: {e}")
    return None


def get_country_from_timezone() -> Optional[str]:
    """
    Infer country code from the system timezone.
    Returns ISO 2-letter country code or None.
    """
    try:
        import datetime
        tz_name = str(datetime.datetime.now().astimezone().tzinfo)

        # Try direct match
        if tz_name in _TZ_TO_COUNTRY:
            cc = _TZ_TO_COUNTRY[tz_name]
            logger.info(f"Country from timezone '{tz_name}': {cc}")
            return cc

        # Try prefix match (e.g., "America/Indiana/Indianapolis" -> "America/Indiana")
        for tz_key, cc in _TZ_TO_COUNTRY.items():
            if tz_name.startswith(tz_key.split("/")[0] + "/" + tz_key.split("/")[1]
                                   if "/" in tz_key else tz_key):
                logger.info(f"Country from timezone prefix '{tz_name}': {cc}")
                return cc

        # Try timezone area mapping
        area = tz_name.split("/")[0] if "/" in tz_name else ""
        area_defaults = {
            "America": "US", "Europe": "DE", "Asia": "CN",
            "Australia": "AU", "Africa": "ZA", "Pacific": "NZ",
        }
        if area in area_defaults:
            cc = area_defaults[area]
            logger.info(f"Country from timezone area '{area}': {cc} (default)")
            return cc
    except Exception as e:
        logger.debug(f"Timezone lookup failed: {e}")
    return None


def detect_country(manual_country_code: Optional[str] = None) -> str:
    """
    Detect the user's country code using a three-tier cascade:
      1. Manual override (if provided)
      2. IP-based geolocation (ip-api.com)
      3. System timezone inference

    Returns ISO 2-letter country code, or "US" as final fallback.
    """
    if manual_country_code:
        cc = manual_country_code.upper().strip()
        logger.info(f"Using manual country code: {cc}")
        return cc

    cc = get_country_from_ip()
    if cc:
        return cc

    cc = get_country_from_timezone()
    if cc:
        return cc

    logger.warning("Could not detect country. Defaulting to 'US'.")
    return "US"


def get_carbon_intensity(
    country_code: Optional[str] = None,
    manual_intensity: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Get grid carbon intensity in gCO2eq/kWh.

    Args:
        country_code: ISO country code.
        manual_intensity: Manual override value.

    Returns:
        Tuple of (intensity_gco2_kwh, source_description).
    """
    if manual_intensity is not None:
        return manual_intensity, "manual override"

    data = _load_json("carbon_intensity.json")

    if country_code:
        cc = country_code.upper().strip()
        countries = data.get("countries", {})
        if cc in countries:
            return countries[cc], f"offline table ({cc})"

    world_avg = data.get("world_average", 494)
    logger.warning(
        f"Carbon intensity not found for '{country_code}'. "
        f"Using world average: {world_avg} gCO2/kWh"
    )
    return world_avg, "world average (country not found)"


def get_electricity_price(
    country_code: Optional[str] = None,
    manual_price: Optional[float] = None,
) -> Tuple[float, str]:
    """
    Get residential electricity price in USD/kWh.

    Args:
        country_code: ISO country code.
        manual_price: Manual override value.

    Returns:
        Tuple of (price_usd_kwh, source_description).
    """
    if manual_price is not None:
        return manual_price, "manual override"

    data = _load_json("electricity_prices.json")

    if country_code:
        cc = country_code.upper().strip()
        countries = data.get("countries", {})
        if cc in countries:
            return countries[cc], f"offline table ({cc})"

    world_avg = data.get("world_average", 0.167)
    logger.warning(
        f"Electricity price not found for '{country_code}'. "
        f"Using world average: ${world_avg}/kWh"
    )
    return world_avg, "world average (country not found)"
