"""Normalize raw card-statement descriptors and derive engineered features."""

from __future__ import annotations

import re

from .data import CITIES

_PROCESSOR = re.compile(r"^(SQ|TST|SP|PY|PAYPAL)\s*\*\s*", re.I)
_REF = re.compile(r"\*[A-Z0-9]{5,}", re.I)  # "*2K4L9X1" style order refs
_PHONE = re.compile(r"\b\d{3}-\d{3}-\d{4}\b")
_STORE = re.compile(r"#?\b0?\d{3,5}\b")
_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO "
    "MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
)
_STATE = re.compile(r"\s(?:" + "|".join(_STATES.split()) + r")$")
_CITY_WORDS = {w for city, _ in CITIES for w in city.split()}
_ONLINE = re.compile(r"\.COM|WWW\.|ONLINE|AMZN|/BILL", re.I)


def clean(descriptor: str) -> str:
    """Strip processor prefixes, order refs, phone/store numbers and location."""
    s = descriptor.upper().strip()
    s = _PROCESSOR.sub("", s)
    s = _REF.sub(" ", s)
    s = _PHONE.sub(" ", s)
    s = _STORE.sub(" ", s)
    s = _STATE.sub("", s.rstrip())
    words = s.replace("WWW.", " ").split()
    # Drop trailing city tokens (including truncated ones like "SAN FRAN")
    while len(words) > 1 and (words[-1] in _CITY_WORDS or any(c.startswith(words[-1]) and len(words[-1]) >= 3 for c in _CITY_WORDS)):
        words.pop()
    if words and words[-1] in {"ONLINE", "WA"}:
        words.pop()
    return " ".join(words) or descriptor.upper().strip()


def features(descriptor: str) -> dict[str, float]:
    """Hand-built signals a statement line carries beyond its words."""
    s = descriptor.upper()
    processor = _PROCESSOR.match(s)
    return {
        "has_processor": float(bool(processor)),
        "processor_square": float(bool(processor) and s.startswith("SQ")),
        "processor_toast": float(bool(processor) and s.startswith("TST")),
        "has_store_number": float(bool(_STORE.search(s))),
        "is_online": float(bool(_ONLINE.search(s) or _REF.search(s))),
        "has_location": float(bool(_STATE.search(s.rstrip()))),
        "digit_ratio": sum(c.isdigit() for c in s) / max(len(s), 1),
        "n_tokens": min(len(clean(s).split()), 6) / 6,
    }
