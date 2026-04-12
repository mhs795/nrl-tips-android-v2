"""
NRL venue weather — fetches real data from Open-Meteo (free, no API key).
Historical: archive-api.open-meteo.com
Forecast:   api.open-meteo.com
"""

import time
from datetime import datetime, date
from functools import lru_cache

import requests

HIST_URL     = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS   = "precipitation_sum,wind_speed_10m_max,temperature_2m_max"

# NRL venue → (latitude, longitude, timezone)
VENUE_COORDS = {
    # NSW
    "accor stadium":            (-33.8469, 151.0638, "Australia/Sydney"),
    "stadium australia":        (-33.8469, 151.0638, "Australia/Sydney"),
    "4 pines park":             (-33.7679, 151.2557, "Australia/Sydney"),
    "brookvale oval":           (-33.7679, 151.2557, "Australia/Sydney"),
    "allianz stadium":          (-33.8908, 151.2249, "Australia/Sydney"),
    "sydney cricket ground":    (-33.8908, 151.2249, "Australia/Sydney"),
    "commbank stadium":         (-33.8120, 151.0038, "Australia/Sydney"),
    "bankwest stadium":         (-33.8120, 151.0038, "Australia/Sydney"),
    "parramatta stadium":       (-33.8120, 151.0038, "Australia/Sydney"),
    "bluebet stadium":          (-33.7508, 150.6921, "Australia/Sydney"),
    "penrith stadium":          (-33.7508, 150.6921, "Australia/Sydney"),
    "leichhardt oval":          (-33.8804, 151.1479, "Australia/Sydney"),
    "pointsbet stadium":        (-34.0436, 151.1206, "Australia/Sydney"),
    "sharks stadium":           (-34.0436, 151.1206, "Australia/Sydney"),
    "netstrata jubilee stadium":(-33.9674, 151.1067, "Australia/Sydney"),
    "win stadium":              (-34.4239, 150.8936, "Australia/Sydney"),
    "mcdonald jones stadium":   (-32.9273, 151.7669, "Australia/Sydney"),
    "hunter stadium":           (-32.9273, 151.7669, "Australia/Sydney"),
    "central coast stadium":    (-33.4316, 151.3428, "Australia/Sydney"),
    # QLD
    "suncorp stadium":          (-27.4649, 153.0095, "Australia/Brisbane"),
    "lang park":                (-27.4649, 153.0095, "Australia/Brisbane"),
    "cbus super stadium":       (-27.9765, 153.3836, "Australia/Brisbane"),
    "robina stadium":           (-27.9765, 153.3836, "Australia/Brisbane"),
    "kayo stadium":             (-26.6501, 153.0652, "Australia/Brisbane"),
    "sunshine coast stadium":   (-26.6501, 153.0652, "Australia/Brisbane"),
    "queensland country bank stadium": (-19.2564, 146.8239, "Australia/Brisbane"),
    "qcb stadium":              (-19.2564, 146.8239, "Australia/Brisbane"),
    "1300smiles stadium":       (-19.2564, 146.8239, "Australia/Brisbane"),
    "bt stadium":               (-19.2564, 146.8239, "Australia/Brisbane"),
    # VIC
    "aami park":                (-37.8230, 144.9844, "Australia/Melbourne"),
    "marvel stadium":           (-37.8167, 144.9472, "Australia/Melbourne"),
    "docklands stadium":        (-37.8167, 144.9472, "Australia/Melbourne"),
    # ACT
    "canberra stadium":         (-35.2436, 149.1285, "Australia/Sydney"),
    "gio stadium":              (-35.2436, 149.1285, "Australia/Sydney"),
    "anz stadium canberra":     (-35.2436, 149.1285, "Australia/Sydney"),
    # NZ
    "go media stadium":         (-36.9042, 174.7776, "Pacific/Auckland"),
    "mt smart stadium":         (-36.9042, 174.7776, "Pacific/Auckland"),
    "eden park":                (-36.8755, 174.7441, "Pacific/Auckland"),
    "hilton hotel auckland":    (-36.9042, 174.7776, "Pacific/Auckland"),
    # SA
    "adelaide oval":            (-34.9158, 138.5961, "Australia/Adelaide"),
    # USA (neutral venues)
    "allegiant stadium":        (36.0909,  -115.1833, "America/Los_Angeles"),
    "sofi stadium":             (33.9535,  -118.3392, "America/Los_Angeles"),
    # Default fallback — Sydney
    "default":                  (-33.8688, 151.2093, "Australia/Sydney"),
}


def _coords(venue: str) -> tuple[float, float, str]:
    """Return (lat, lon, timezone) for a venue name, falling back to Sydney."""
    key = venue.lower().strip()
    # Exact match
    if key in VENUE_COORDS:
        return VENUE_COORDS[key]
    # Partial match
    for name, coords in VENUE_COORDS.items():
        if name in key or key in name:
            return coords
    return VENUE_COORDS["default"]


def _fetch(url: str, params: dict) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_historical_weather(venue: str, game_date: str) -> dict:
    """
    Fetch actual weather for a past game.
    game_date: 'YYYY-MM-DD'
    Returns: {temp_c, rain_mm, wind_kmh}
    """
    lat, lon, tz = _coords(venue)
    data = _fetch(HIST_URL, {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": game_date,
        "end_date":   game_date,
        "daily":      DAILY_VARS,
        "timezone":   tz,
    })
    return _extract(data, 0)


def get_forecast_weather(venue: str, game_date: str) -> dict:
    """
    Fetch forecast weather for an upcoming game.
    game_date: 'YYYY-MM-DD'
    Returns: {temp_c, rain_mm, wind_kmh}
    """
    lat, lon, tz = _coords(venue)
    data = _fetch(FORECAST_URL, {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         DAILY_VARS,
        "timezone":      tz,
        "forecast_days": 14,
    })
    if not data:
        return _default()
    # Find the index matching game_date
    times = data.get("daily", {}).get("time", [])
    try:
        idx = times.index(game_date)
        return _extract(data, idx)
    except ValueError:
        return _default()


def get_weather(venue: str, game_date: str) -> dict:
    """Auto-select historical vs forecast based on date."""
    if not game_date:
        return _default()
    try:
        gd = datetime.strptime(game_date, "%Y-%m-%d").date()
    except ValueError:
        return _default()
    if gd < date.today():
        return get_historical_weather(venue, game_date)
    else:
        return get_forecast_weather(venue, game_date)


def _extract(data: dict | None, idx: int) -> dict:
    if not data:
        return _default()
    try:
        d = data["daily"]
        return {
            "weather_temp_c":   round(d["temperature_2m_max"][idx] or 20, 1),
            "weather_rain_mm":  round(d["precipitation_sum"][idx] or 0,  1),
            "weather_wind_kmh": round(d["wind_speed_10m_max"][idx] or 15, 1),
        }
    except (KeyError, IndexError, TypeError):
        return _default()


def _default() -> dict:
    return {"weather_temp_c": 20.0, "weather_rain_mm": 0.0, "weather_wind_kmh": 15.0}


def batch_historical(venue_dates: list[tuple[str, str]]) -> dict[tuple, dict]:
    """
    Fetch weather for many (venue, date) pairs efficiently.
    Groups by venue to minimise API calls, fetching a date range per venue.
    Returns dict keyed by (venue, date).
    """
    from collections import defaultdict

    # Group dates by (lat, lon, tz)
    by_coords: dict[tuple, list] = defaultdict(list)
    coord_map: dict[tuple, str] = {}
    for venue, gdate in venue_dates:
        coords = _coords(venue)
        by_coords[coords].append(gdate)
        coord_map[coords] = venue

    results = {}
    for (lat, lon, tz), dates in by_coords.items():
        if not dates:
            continue
        dates_sorted = sorted(set(d for d in dates if d))
        if not dates_sorted:
            continue
        start, end = dates_sorted[0], dates_sorted[-1]
        data = _fetch(HIST_URL, {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": start,
            "end_date":   end,
            "daily":      DAILY_VARS,
            "timezone":   tz,
        })
        if not data:
            for d in dates:
                results[(coord_map[(lat, lon, tz)], d)] = _default()
            continue
        times = data.get("daily", {}).get("time", [])
        for gdate in dates:
            try:
                idx = times.index(gdate)
                w = _extract(data, idx)
            except ValueError:
                w = _default()
            results[(coord_map[(lat, lon, tz)], gdate)] = w
        time.sleep(0.2)   # polite rate limiting

    return results
