"""
COD order decision engine.

For one order it works out the expected profit (in rupees) of each action and
picks the best one that the guardrails in config/threshold.yaml allow.

  p = chance this order is returned if we ship it as plain COD (from the risk model)

Actions: ship, confirm (WhatsApp), nudge (offer a prepaid discount), cancel.
A separate "hold" is used when the address check says a person must look first.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import yaml

SHIP, CONFIRM, NUDGE, CANCEL, HOLD = "ship", "confirm", "nudge", "cancel", "hold"

CONFIG_PATH = Path(
    os.environ.get(
        "THRESHOLD_CONFIG", Path(__file__).parent / "config" / "threshold.yaml"
    )
)


# ---------------------------------------------------------------- config ----
_REQUIRED = [  # (section, key, lowest allowed, highest allowed)
    ("economics", "gross_margin_pct", 0, 1),
    ("economics", "forward_shipping", 0, None),
    ("economics", "return_shipping", 0, None),
    ("economics", "handling", 0, None),
    ("economics", "damage_rate", 0, 1),
    ("economics", "product_cost", 0, None),
    ("economics", "prepaid_rto_rate", 0, 1),
    ("economics", "repeat_value_first_time", 0, None),
    ("confirmation", "genuine_confirm_rate", 0, 1),
    ("confirmation", "returner_confirm_rate", 0, 1),
    ("confirmation", "message_cost", 0, None),
    ("nudge", "accept_base_genuine", None, None),
    ("nudge", "accept_base_returner", None, None),
    ("nudge", "discount_elasticity", 0, None),
    ("address_quality", "hold_manual_below", 0, 1),
    ("address_quality", "force_confirm_below", 0, 1),
    ("address_quality", "default_when_missing", 0, 1),
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for section, key, lo, hi in _REQUIRED:
        try:
            v = cfg[section][key]
        except (KeyError, TypeError):
            raise ValueError(f"threshold.yaml: '{section} -> {key}' is missing")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(
                f"threshold.yaml: '{section} -> {key}' must be a number, got {v!r}"
            )
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(
                f"threshold.yaml: '{section} -> {key}' = {v} is outside the allowed range"
            )
    if not cfg["nudge"].get("discount_grid"):
        raise ValueError(
            "threshold.yaml: 'nudge -> discount_grid' needs at least one discount"
        )
    for key in ("ship_max_p", "nudge_max_p", "confirm_max_p"):
        for seg in ("first_time", "repeat"):
            v = cfg["bands"][key][seg]
            if not 0 <= v <= 1:
                raise ValueError(
                    f"threshold.yaml: 'bands -> {key} -> {seg}' must be between 0 and 1"
                )
    return cfg


CFG = load_config()
E = CFG["economics"]


# ----------------------------------------------------------------- money ----
def return_cost() -> float:
    """Rupees lost on one returned COD order: both shipping legs, handling, and
    the damaged share of the product."""
    return (
        E["forward_shipping"]
        + E["return_shipping"]
        + E["handling"]
        + E["damage_rate"] * E["product_cost"]
    )


def margin(order_value: float, is_first_time: bool) -> float:
    """Rupees earned if this order is delivered. First-time buyers also carry their expected repeat value"""
    m = E["gross_margin_pct"] * order_value
    return m + (E["repeat_value_first_time"] if is_first_time else 0.0)


def accept_prob(kind: str, d: float, order_value: float) -> float:
    """Chance a customer agrees to prepay for discount d in rupees
    It uses sigmoid-based acceptance probability model
    """
    n = CFG["nudge"]
    x = n[f"accept_base_{kind}"] + n["discount_elasticity"] * (d / order_value)
    return 1 / (1 + math.exp(-x))


def confirm_stats(p: float) -> tuple[float, float]:
    """(share of all orders that get confirmed and shipped, return rate among them).
    Both come from the same two assumptions, so they can never contradict each other."""
    g = CFG["confirmation"]["genuine_confirm_rate"]
    b = CFG["confirmation"]["returner_confirm_rate"]
    shipped = p * b + (1 - p) * g
    return shipped, (p * b / shipped if shipped else 0.0)


def expected_rto(
    action: str, p: float, d: float = 0.0, order_value: float = 649.0
) -> float:
    """Share of ALL orders (not just shipped ones) that end up returned. Used for the dashboard."""
    b = CFG["confirmation"]["returner_confirm_rate"]
    r = E["prepaid_rto_rate"]
    if action == SHIP:
        return p
    if action == CONFIRM:
        return p * b
    if action == NUDGE:
        ag = accept_prob("genuine", d, order_value)
        ab = accept_prob("returner", d, order_value)
        return (1 - p) * ag * r + p * (ab * r + (1 - ab) * b)
    return 0.0  # cancel / hold: nothing ships


def ev(
    action: str, p: float, d: float, order_value: float, is_first_time: bool
) -> float:
    """Expected profit in rupees of one action on one order."""
    M, Lc = margin(order_value, is_first_time), return_cost()
    k = CFG["confirmation"]["message_cost"]
    g = CFG["confirmation"]["genuine_confirm_rate"]
    b = CFG["confirmation"]["returner_confirm_rate"]

    if action == SHIP:
        return (1 - p) * M - p * Lc

    if action == CONFIRM:
        # genuine customers confirm at g and deliver; returners confirm at b and come back
        return (1 - p) * g * M - p * b * Lc - k

    if action == NUDGE:
        r = E["prepaid_rto_rate"]
        ag = accept_prob("genuine", d, order_value)
        ab = accept_prob("returner", d, order_value)
        prepaid = (1 - r) * (M - d) - r * Lc  # value of an order that pays online
        genuine = (
            ag * prepaid + (1 - ag) * g * M
        )  # decliners fall back to WhatsApp confirmation
        returner = ab * prepaid - (1 - ab) * b * Lc
        return (1 - p) * genuine + p * returner - k

    return 0.0  # cancel


def optimal_discount(p: float, order_value: float, is_first_time: bool) -> float:
    """The discount from the grid that gives the best nudge value for this order."""
    return max(
        CFG["nudge"]["discount_grid"],
        key=lambda d: ev(NUDGE, p, d, order_value, is_first_time),
    )


# -------------------------------------------------------------- decision ----
class Decision(NamedTuple):
    action: str
    discount: float
    reason: str


def _rs(x: float) -> str:
    return f"-₹{abs(x):,.0f}" if round(x) < 0 else f"₹{x:,.0f}"


def decide(
    p: float,
    addr_quality: Optional[float],
    order_value: float,
    is_first_time: bool = False,
    drivers: Optional[Sequence[str]] = None,
) -> Decision:
    """
    p             chance of return if shipped as COD (0 to 1)
    addr_quality  0 to 1 score from the address/name/phone check, or None if it failed
    order_value   order total in rupees
    is_first_time True if the customer has never ordered before
    drivers       optional short phrases for why the risk is high, e.g. ["tier-3 city", "junk address"]
    returns       Decision(action, discount in rupees, plain-English reason)
    """
    if not 0 <= p <= 1:
        raise ValueError(f"p must be between 0 and 1, got {p}")
    if not order_value or order_value <= 0:
        return Decision(
            HOLD,
            0,
            "Order value is zero or missing. Check the order before doing anything.",
        )

    # 1. hard overrides from the address check, before any money maths
    aq = CFG["address_quality"]
    note = ""
    if addr_quality is None or (
        isinstance(addr_quality, float) and math.isnan(addr_quality)
    ):
        addr_quality = aq["default_when_missing"]
        note = " The address check was unavailable, so a neutral score was used."
    if addr_quality < aq["hold_manual_below"]:
        return Decision(
            HOLD,
            0,
            f"Address, name or phone look unreliable (score {addr_quality:.2f}). "
            f"Held for a person to review before anything ships.{note}",
        )
    if addr_quality < aq["force_confirm_below"]:
        return Decision(
            CONFIRM,
            0,
            f"Address, name or phone look doubtful (score {addr_quality:.2f}). "
            f"The customer must confirm on WhatsApp before we ship.{note}",
        )

    # 2. expected profit of every action
    d = optimal_discount(p, order_value, is_first_time)
    evs = {
        SHIP: ev(SHIP, p, 0, order_value, is_first_time),
        CONFIRM: ev(CONFIRM, p, 0, order_value, is_first_time),
        NUDGE: ev(NUDGE, p, d, order_value, is_first_time),
        CANCEL: 0.0,
    }
    pure_best = max(evs, key=evs.get)

    # 3. guardrails: each action only up to its risk ceiling; first-timers never auto-cancelled
    bands = CFG["bands"]
    seg = "first_time" if is_first_time else "repeat"
    ceiling = {
        SHIP: bands["ship_max_p"][seg],
        NUDGE: bands["nudge_max_p"][seg],
        CONFIRM: bands["confirm_max_p"][seg],
    }
    allowed = {a: v for a, v in evs.items() if a == CANCEL or p <= ceiling[a]}
    if is_first_time and bands["never_cancel_first_time"]:
        allowed.pop(CANCEL, None)
    if not allowed:
        allowed = {CONFIRM: evs[CONFIRM]}

    action = max(allowed, key=allowed.get)
    disc = d if action == NUDGE else 0

    # 4. plain-English reason
    who = "first-time customer" if is_first_time else "repeat customer"
    parts = [
        f"{p:.0%} chance this order is returned. {who.capitalize()}, order ₹{order_value:,.0f}.",
        f"Expected profit: ship as-is {_rs(evs[SHIP])}, WhatsApp confirmation {_rs(evs[CONFIRM])}, "
        f"prepaid offer of ₹{d:.0f} off {_rs(evs[NUDGE])}.",
    ]
    parts.append(
        {
            SHIP: "Shipping as-is is the best option.",
            CONFIRM: "Ask the customer to confirm on WhatsApp before shipping.",
            NUDGE: f"Offer ₹{d:.0f} off to pay online. If they decline, fall back to WhatsApp confirmation.",
            CANCEL: "Every other option loses money, so cancel.",
        }[action]
    )
    if action != pure_best:
        parts.append(
            f"The pure money maths preferred '{pure_best}', but a guardrail in the config ruled it out."
        )
    if drivers:
        parts.append("Main risk signals: " + ", ".join(drivers) + ".")
    return Decision(action, disc, " ".join(parts) + note)


# ------------------------------------------------------------ self-check ----
def threshold_table(
    order_value: float = 649, is_first_time: bool = False
) -> list[tuple[float, str, float]]:
    """Where the engine switches action as risk rises, for one order size."""
    rows = []
    for i in range(0, 20):
        p = i / 20
        dec = decide(p, 1.0, order_value, is_first_time)
        rows.append((p, dec.action, dec.discount))
    return rows


if __name__ == "__main__":
    print(f"Cost of one return: ₹{return_cost():.1f}")
    for first in (False, True):
        for ov in (649, 919):
            print(f"\n{'First-time' if first else 'Repeat'} customer, order ₹{ov}:")
            prev = None
            for p, act, disc in threshold_table(ov, first):
                label = f"{act}" + (f" (₹{disc:.0f} off)" if act == NUDGE else "")
                if label != prev:
                    print(f"  from risk {p:>4.0%}: {label}")
                    prev = label
    print("\nExample:")
    print(decide(0.42, 0.9, 649, True, ["tier-3 city", "first order"]).reason)
