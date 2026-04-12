"""
NRL travel distances — computes km each team travels to reach the venue.
Uses Haversine formula (great-circle distance).
Home team travel = 0.  Away team = home city → venue.
"""

import math

# Team home city coordinates (their primary home ground)
TEAM_HOME = {
    "Brisbane Broncos":            (-27.4649, 153.0095),   # Suncorp Stadium
    "Canberra Raiders":            (-35.2436, 149.1285),   # GIO Stadium
    "Canterbury Bulldogs":         (-33.8120, 151.0038),   # CommBank Stadium
    "Cronulla Sharks":             (-34.0436, 151.1206),   # PointsBet Stadium
    "Gold Coast Titans":           (-27.9765, 153.3836),   # Cbus Super Stadium
    "Manly Sea Eagles":            (-33.7679, 151.2557),   # 4 Pines Park
    "Melbourne Storm":             (-37.8230, 144.9844),   # AAMI Park
    "Newcastle Knights":           (-32.9273, 151.7669),   # McDonald Jones Stadium
    "New Zealand Warriors":        (-36.9042, 174.7776),   # Go Media Stadium
    "North Queensland Cowboys":    (-19.2564, 146.8239),   # QCB Stadium
    "Parramatta Eels":             (-33.8120, 151.0038),   # CommBank Stadium
    "Penrith Panthers":            (-33.7508, 150.6921),   # BlueBet Stadium
    "South Sydney Rabbitohs":      (-33.8908, 151.2249),   # Allianz Stadium
    "St George Illawarra Dragons": (-33.9674, 151.1067),   # Netstrata Jubilee
    "Sydney Roosters":             (-33.8908, 151.2249),   # Allianz Stadium
    "Wests Tigers":                (-33.8804, 151.1479),   # Leichhardt Oval
    "Dolphins":                    (-27.4649, 153.0095),   # Suncorp (share with Broncos)
}

# Venue name → coordinates (reuse from weather.py dict but keep independent)
VENUE_COORDS = {
    "accor stadium":            (-33.8469, 151.0638),
    "stadium australia":        (-33.8469, 151.0638),
    "4 pines park":             (-33.7679, 151.2557),
    "brookvale oval":           (-33.7679, 151.2557),
    "allianz stadium":          (-33.8908, 151.2249),
    "sydney cricket ground":    (-33.8908, 151.2249),
    "commbank stadium":         (-33.8120, 151.0038),
    "bankwest stadium":         (-33.8120, 151.0038),
    "parramatta stadium":       (-33.8120, 151.0038),
    "bluebet stadium":          (-33.7508, 150.6921),
    "penrith stadium":          (-33.7508, 150.6921),
    "leichhardt oval":          (-33.8804, 151.1479),
    "pointsbet stadium":        (-34.0436, 151.1206),
    "sharks stadium":           (-34.0436, 151.1206),
    "netstrata jubilee stadium":(-33.9674, 151.1067),
    "jubilee stadium":          (-33.9674, 151.1067),
    "win stadium":              (-34.4239, 150.8936),
    "mcdonald jones stadium":   (-32.9273, 151.7669),
    "hunter stadium":           (-32.9273, 151.7669),
    "central coast stadium":    (-33.4316, 151.3428),
    "industree group stadium":  (-33.4316, 151.3428),
    "campbelltown sports stadium": (-34.0667, 150.8167),
    "belmore sports ground":    (-33.9106, 151.0833),
    "suncorp stadium":          (-27.4649, 153.0095),
    "lang park":                (-27.4649, 153.0095),
    "cbus super stadium":       (-27.9765, 153.3836),
    "robina stadium":           (-27.9765, 153.3836),
    "kayo stadium":             (-26.6501, 153.0652),
    "sunshine coast stadium":   (-26.6501, 153.0652),
    "queensland country bank stadium": (-19.2564, 146.8239),
    "qcb stadium":              (-19.2564, 146.8239),
    "1300smiles stadium":       (-19.2564, 146.8239),
    "bt stadium":               (-19.2564, 146.8239),
    "aami park":                (-37.8230, 144.9844),
    "marvel stadium":           (-37.8167, 144.9472),
    "gio stadium":              (-35.2436, 149.1285),
    "canberra stadium":         (-35.2436, 149.1285),
    "go media stadium":         (-36.9042, 174.7776),
    "mt smart stadium":         (-36.9042, 174.7776),
    "eden park":                (-36.8755, 174.7441),
    "sky stadium":              (-41.2865, 174.7762),   # Wellington
    "allegiant stadium":        (36.0909,  -115.1833),
    "sofi stadium":             (33.9535,  -118.3392),
    "adelaide oval":            (-34.9158, 138.5961),
    "optus stadium":            (-31.9505, 115.8605),
    "hbf park":                 (-31.9505, 115.8605),
    "townsville":               (-19.2564, 146.8239),
    "browne park":              (-23.3772, 150.5120),   # Rockhampton
    "bb print stadium":         (-21.1411, 149.1633),   # Mackay
    "barlow park":              (-16.9186, 145.7781),   # Cairns
    "scully park":              (-31.0840, 150.9290),   # Tamworth
    "carrington park":          (-33.4127, 149.5782),   # Bathurst
    "glen willow oval":         (-32.2569, 148.6478),   # Mudgee
    "apex oval":                (-29.6879, 148.1125),   # Bourke area
    "mcdonalds park":           (-29.6879, 148.1125),
    "polytec stadium":          (-37.6940, 144.3327),   # Ballarat
    "fmg stadium waikato":      (-37.7870, 175.2793),   # Hamilton NZ
    "apollo projects stadium":  (-43.5320, 172.6362),   # Christchurch NZ
    "mclean park":              (-39.4928, 176.9120),   # Napier NZ
}


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two points."""
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ/2)**2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def _venue_coords(venue: str) -> tuple[float, float] | None:
    key = venue.lower().strip()
    if key in VENUE_COORDS:
        return VENUE_COORDS[key]
    for name, coords in VENUE_COORDS.items():
        if name in key or key in name:
            return coords
    return None


def travel_km(team: str, venue: str) -> float:
    """
    Returns the km a team travels to reach the venue.
    Returns 0 if team or venue not found (treated as home game).
    """
    home = TEAM_HOME.get(team)
    dest = _venue_coords(venue)
    if not home or not dest:
        return 0.0
    dist = _haversine(home[0], home[1], dest[0], dest[1])
    # Under ~30 km = effectively home ground, call it 0
    return 0.0 if dist < 30 else dist
