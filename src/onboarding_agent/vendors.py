"""Mock vendor systems plus the adapters that map their raw payloads onto our typed models.

- ReferenceDataVendor: instrument static data, vendor style field names.
- PricingVendor: end of day prices from a primary and a secondary source.
- BookingSystem: the system of record we write to (persisted to JSON so approvals can resume in another process).

Fault injection (see vendor_seed.json) reproduces the failures the agent has to handle:
transient timeouts, permanent timeouts, and a vendor silently changing a field's format.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import PACKAGE_DIR
from .models import InstrumentRecord, PriceQuote, PriceSource

SEED_FILE = PACKAGE_DIR / "data" / "vendor_seed.json"

_TYPE_MAP = {"GOVT": "government_bond", "CORP": "corporate_bond", "FRN": "frn"}


class VendorError(Exception):
    """A non-retryable data problem (not found, rejected payload)."""


class NotFoundError(VendorError):
    pass


class VendorTimeout(Exception):
    """A transient failure. Safe to retry."""


def _simulate_latency() -> None:
    """Vendor calls are not free. AGENT_VENDOR_LATENCY_MS sets the mean (0 disables it, as in tests)."""
    mean_ms = float(os.getenv("AGENT_VENDOR_LATENCY_MS", "60"))
    if mean_ms > 0:
        time.sleep(random.lognormvariate(0, 0.5) * mean_ms / 1000)


def load_seed() -> dict[str, Any]:
    return json.loads(SEED_FILE.read_text())


class ReferenceDataVendor:
    def __init__(self, seed: dict[str, Any]):
        self._data = seed["reference_data"]
        self._faults = {k: v.get("reference_data") for k, v in seed["faults"].items()}
        self._calls: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def get_security(self, isin: str) -> dict[str, Any]:
        _simulate_latency()
        with self._lock:
            self._calls[isin] += 1
            call_no = self._calls[isin]
        fault = self._faults.get(isin)
        if fault == "always_timeout" or (fault == "transient_once" and call_no == 1):
            raise VendorTimeout(f"reference data request for {isin} timed out")
        if isin not in self._data:
            raise NotFoundError(f"{isin} not found in reference data")
        return dict(self._data[isin])


class PricingVendor:
    def __init__(self, seed: dict[str, Any], now: datetime | None = None):
        self._prices = seed["prices"]
        self._faults = seed["faults"]
        self._now = now

    def get_eod_price(self, isin: str, source: PriceSource) -> dict[str, Any]:
        _simulate_latency()
        quotes = self._prices.get(isin)
        if not quotes:
            raise NotFoundError(f"no {source} price for {isin}")
        q = quotes[source]
        px = q["px"]
        if self._faults.get(isin, {}).get(f"{source}_prices") == "format_drift":
            # The vendor started quoting as a fraction of par (0.9915) instead of percent of par (99.15).
            # The payload is still well formed, which is what makes this failure dangerous.
            px = round(px / 100, 6)
        now = self._now or datetime.now(timezone.utc)
        as_of = now - timedelta(hours=q["age_h"])
        return {"ISIN": isin, "SRC": source, "PX_CLEAN": px, "PX_TIME": as_of.isoformat()}


class BookingSystem:
    """System of record. Writes only happen after validation and human approval."""

    _lock = threading.Lock()

    def __init__(self, path: Path, seed: dict[str, Any]):
        self._path = path
        if not path.exists():
            self._save({"instruments": {i: {"isin": i} for i in seed["booked"]}, "prices": {}, "breaks": []})

    def _load(self) -> dict[str, Any]:
        return json.loads(self._path.read_text())

    def _save(self, state: dict[str, Any]) -> None:
        self._path.write_text(json.dumps(state, indent=2, default=str))

    def exists(self, isin: str) -> bool:
        return isin in self._load()["instruments"]

    def create_instrument(self, record: InstrumentRecord) -> str:
        with self._lock:
            state = self._load()
            if record.isin in state["instruments"]:
                raise VendorError(f"{record.isin} already exists in the booking system")
            state["instruments"][record.isin] = record.model_dump(mode="json")
            self._save(state)
        return f"INS-{record.isin}"

    def publish_price(self, isin: str, price: float, as_of: str) -> str:
        with self._lock:
            state = self._load()
            state["prices"][isin] = {"clean_price": price, "as_of": as_of}
            self._save(state)
        return f"PX-{isin}"

    def flag_break(self, isin: str, details: dict[str, Any]) -> str:
        with self._lock:
            state = self._load()
            state["breaks"].append({"isin": isin, **details})
            self._save(state)
            n = len(state["breaks"])
        return f"BRK-{n:05d}"

    def snapshot(self) -> dict[str, Any]:
        return self._load()


# Adapters: vendor payload -> typed model. Validation errors here are real data problems, not formatting noise.

def _vendor_date(s: str) -> str:
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def adapt_security(raw: dict[str, Any]) -> InstrumentRecord:
    return InstrumentRecord(
        isin=raw["ID_ISIN"],
        name=raw["SECURITY_DES"],
        instrument_type=_TYPE_MAP.get(raw["SECURITY_TYP"], raw["SECURITY_TYP"]),
        issuer=raw["ISSUER"],
        currency=raw["CRNCY"],
        coupon_rate=raw["CPN"],
        issue_date=_vendor_date(raw["ISSUE_DT"]),
        maturity_date=_vendor_date(raw["MATURITY"]),
        day_count=raw["DAY_CNT_DES"],
    )


def adapt_price(raw: dict[str, Any]) -> PriceQuote:
    return PriceQuote(isin=raw["ISIN"], source=raw["SRC"], clean_price=raw["PX_CLEAN"], as_of=raw["PX_TIME"])


class Services:
    """Everything the tools talk to. Tests build their own with a temp directory."""

    def __init__(self, data_dir: Path, now: datetime | None = None):
        seed = load_seed()
        self.reference = ReferenceDataVendor(seed)
        self.pricing = PricingVendor(seed, now=now)
        self.booking = BookingSystem(data_dir / "booking_system.json", seed)
