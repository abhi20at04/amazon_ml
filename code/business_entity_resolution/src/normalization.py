"""Generic normalization and compact blocking-key extraction."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Terms that rarely distinguish businesses or locations. Deliberately language-agnostic
# apart from common corporate/legal terms; country labels are never enumerated.
STOPWORDS = {
    "and", "the", "of", "at", "in", "near", "opposite", "beside", "behind",
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive",
    "dr", "highway", "hwy", "route", "unit", "suite", "ste", "floor", "fl",
    "building", "bldg", "block", "sector", "district", "city", "town", "village",
    "county", "state", "corp", "corporation",
    "inc", "incorporated", "llc", "ltd", "limited", "private", "pvt", "company",
    "co", "plc", "llp", "sarl", "sas", "gmbh", "bv", "the",
}

ABBREVIATIONS = {
    "corporation": "corp", "incorporated": "inc", "limited": "ltd",
    "private": "pvt", "company": "co", "street": "st", "road": "rd",
    "avenue": "ave", "boulevard": "blvd", "drive": "dr", "lane": "ln",
    "highway": "hwy", "apartment": "apt", "suite": "ste", "north": "n",
    "south": "s", "east": "e", "west": "w", "centre": "center",
}


def normalize(value: object) -> str:
    """Lowercase, remove diacritics/punctuation, and canonicalize common variants."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = [ABBREVIATIONS.get(token, token) for token in text.split()]
    # Normalize common legal suffixes without dropping them from similarity features.
    return " ".join(tokens)


def informative_tokens(value: object, *, field: str, limit: int = 5) -> list[str]:
    """Return a small, stable set of informative word keys for a field."""
    return _informative_tokens_normalized(normalize(value), field=field, limit=limit)


def _informative_tokens_normalized(text: str, *, field: str, limit: int) -> list[str]:
    tokens = text.split()
    unique = {
        token for token in tokens
        if token not in STOPWORDS and (len(token) >= 3 or token.isdigit())
    }
    # Numeric address components (house numbers, postal codes) are especially useful;
    # otherwise prefer longer words, which tend to be more discriminative.
    if field == "address":
        ordered = sorted(unique, key=lambda t: (not (t.isdigit() and len(t) >= 3), -len(t), t))
    else:
        ordered = sorted(unique, key=lambda t: (-len(t), t))
    return ordered[:limit]


def blocking_keys_normalized(name: str, address: str, country: str, *, limit: int = 3) -> list[str]:
    """Build blocking keys from fields already processed by :func:`normalize`."""
    country_key = country
    keys = [f"n|{country_key}|{token}" for token in _informative_tokens_normalized(name, field="name", limit=limit)]
    keys.extend(f"a|{country_key}|{token}" for token in _informative_tokens_normalized(address, field="address", limit=2))
    return list(dict.fromkeys(keys))


def blocking_keys(name: object, address: object, country: object, *, limit: int = 3) -> list[str]:
    """Build country-scoped name/address token keys; works with unseen country labels."""
    return blocking_keys_normalized(normalize(name), normalize(address), normalize(country), limit=limit)


def token_set(value: object) -> set[str]:
    return {t for t in normalize(value).split() if len(t) > 1}


@lru_cache(maxsize=50000)
def token_set_normalized(value: str) -> frozenset[str]:
    """Token set for a string that has already been normalized."""
    return frozenset(token for token in value.split() if len(token) > 1)


def normalized_record(record: dict[str, str]) -> dict[str, str]:
    """Return a shallow copy with text fields normalized exactly once."""
    result = dict(record)
    for field in ("business_name", "business_address", "country"):
        result[field] = normalize(record.get(field, ""))
    return result
