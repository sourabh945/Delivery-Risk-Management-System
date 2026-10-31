"""
Train the RTO risk model on past COD orders.

Output: outputs/model.pkl (model + everything needed to score a NEW order)
        outputs/orders_scored.csv (every order, its features, its score p, and which split it is in)

p is the chance an order is returned if shipped as plain COD. The decision engine
treats p as a real probability, so it has to be calibrated, not just well ranked.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/orders.csv")
OUT = Path("outputs")
TRAIN_END = "2026-08-01"    # train on orders before this date
TEST_START = "2026-08-08"   # test on orders from here (7-day gap: recent outcomes are not settled yet)
SMOOTH = 20                 # pulls small groups toward the overall rate (for state only)

METRO = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata"]
TIER2 = ["Jaipur", "Lucknow", "Indore", "Guwahati", "Patna"]
JUNK_NAMES = {"test", "test order", "qwerty", "asdf", "abc", "aaa", "na", "xyz", "xyz xyz"}
JUNK_ADDRESSES = {"xyz", "call me", ".", "na", "asdf", "test"}

FEATS = [
    "state_risk", "tier2", "tier3",                    # where
    "addr_len", "has_house", "has_landmark", "junk_any", "phone_valid",   # how clean the order details are
    "is_first", "prior_rto1", "prior_rto2",            # who
    "cod_amount", "hi_value", "num_items",             # what
]


# ------------------------------------------------------------- loading ------
def load_orders(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"phone": str})
    df = df.drop_duplicates().copy()                         # the file has repeated rows
    df["order_datetime"] = pd.to_datetime(df["order_datetime"])
    df["status"] = df["final_status"].str.strip().str.upper()
    df = df[df["status"].isin(["DELIVERED", "RTO"])].copy()  # drop in-transit orders
    df["y"] = (df["status"] == "RTO").astype(int)
    return df.sort_values("order_datetime", kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------ features ------
def smoothed_rate(returns, n, prior: float, m: int = SMOOTH):
    return np.clip((returns + m * prior) / (n + m), 0, 1)   # works for a column and for one number


def build_features(df: pd.DataFrame, prior: float, state_rates: dict | None = None) -> pd.DataFrame:
    """Same code for training and for scoring a new order.
    Training: state_risk uses only EARLIER orders (state_rates=None, df sorted by time).
    Live: state_risk is looked up in the saved table."""
    df = df.copy()

    if state_rates is None:
        gb = df.groupby("state")["y"]
        df["state_risk"] = smoothed_rate(gb.cumsum() - df["y"], gb.cumcount(), prior)
    else:
        df["state_risk"] = df["state"].map(state_rates)
    df["state_risk"] = df["state_risk"].fillna(prior)

    df["tier2"] = df["city"].isin(TIER2).astype(int)
    df["tier3"] = (df["city"].notna() & ~df["city"].isin(METRO + TIER2)).astype(int)

    addr = df["address"].fillna("").astype(str).str.strip()
    name = df["customer_name"].fillna("").astype(str).str.strip()
    df["addr_len"] = addr.str.len()
    df["has_house"] = addr.str.contains(r"\d").astype(int)
    df["has_landmark"] = addr.str.contains(r"near|ke paas|ke saamne|gali|chowk", case=False).astype(int)
    junk_addr = addr.str.lower().isin(JUNK_ADDRESSES) | (addr.str.len() < 8)
    junk_name = name.str.lower().isin(JUNK_NAMES) | (name.str.split().str.len() == 1)
    df["junk_any"] = (junk_addr | junk_name).astype(int)

    digits = df["phone"].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    digits = digits.where(~((digits.str.len() == 12) & digits.str.startswith("91")), digits.str[2:])
    df["phone_valid"] = ((digits.str.len() == 10) & digits.str[0].isin(list("6789"))).astype(int)

    df["is_first"] = (df["customer_prior_orders"] == 0).astype(int)
    df["prior_rto1"] = (df["customer_prior_rto"] == 1).astype(int)   # one earlier return
    df["prior_rto2"] = (df["customer_prior_rto"] >= 2).astype(int)   # repeat offenders: separate, much stronger flag

    ov = df["order_value"].where(df["order_value"] > 0)              # a Rs 0 order is bad data, not a cheap order
    df["cod_amount"] = ov.clip(upper=5000)                           # fixed cap (not learned from the data)
    df["hi_value"] = (ov > 1500).astype(int)                         # returns jump above Rs 1,500; a straight line can't see that
    return df


def make_model() -> object:
    # no class_weight: it would push predicted risk to about 2x the real rate and break every threshold
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000),
    )


# ------------------------------------------------------------ scoring -------
def predict_order(order: dict, bundle: dict) -> float:
    """Risk p for one new order. Needs: state, city, address, customer_name, phone,
    customer_prior_orders, customer_prior_rto, order_value, num_items."""
    rates = {
        s: float(smoothed_rate(r, n, bundle["prior"], bundle["smooth"]))
        for s, (r, n) in bundle["state_counts"].items()
    }
    X = build_features(pd.DataFrame([order]), bundle["prior"], rates)
    return float(bundle["model"].predict_proba(X[bundle["features"]])[:, 1][0])


# --------------------------------------------------------------- main -------
def main() -> None:
    df = load_orders()
    train_mask = df["order_datetime"] < TRAIN_END
    test_mask = df["order_datetime"] >= TEST_START
    prior = df.loc[train_mask, "y"].mean()          # overall rate from training data only
    df = build_features(df, prior)

    train, test = df[train_mask], df[test_mask]
    model = make_model().fit(train[FEATS], train["y"])
    p = model.predict_proba(test[FEATS])[:, 1]

    print(f"rows {len(df)} | train {len(train)} | test {len(test)}")
    print(f"AUC {roc_auc_score(test['y'], p):.3f} | PR-AUC {average_precision_score(test['y'], p):.3f}")
    base = brier_score_loss(test["y"], np.full(len(test), prior))
    print(f"Brier {brier_score_loss(test['y'], p):.4f} (always guessing {prior:.0%}: {base:.4f}; lower is better)")
    print(f"average predicted risk {p.mean():.1%} vs actual {test['y'].mean():.1%}")
    if abs(p.mean() - test["y"].mean()) > 0.05:
        print("WARNING: predicted risk is off by more than 5 points. Thresholds will be wrong until this is fixed.")

    bands = pd.cut(p, [0, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0])
    print("\nrisk band    orders  predicted  actual")
    for b, g in test.assign(p=p, band=bands).groupby("band", observed=True):
        print(f"{str(b):12s} {len(g):6d}  {g['p'].mean():8.1%}  {g['y'].mean():6.1%}")

    # final model: refit on every settled order before saving (the test above was only a check)
    final = make_model().fit(df[FEATS], df["y"])
    state_counts = {
        s: (int(g["y"].sum()), int(len(g))) for s, g in df.groupby("state")
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "model.pkl", "wb") as f:
        pickle.dump({"model": final, "features": FEATS, "prior": float(prior), "smooth": SMOOTH,
                     "state_counts": state_counts}, f)

    # scores from the check model. Use only split == 'test' rows for backtests: train rows are in-sample.
    df["p"] = model.predict_proba(df[FEATS])[:, 1]
    df["split"] = np.where(train_mask, "train", np.where(test_mask, "test", "gap"))
    df.to_csv(OUT / "orders_scored.csv", index=False)
    print(f"\nsaved {OUT/'model.pkl'} and {OUT/'orders_scored.csv'}")


if __name__ == "__main__":
    main()
