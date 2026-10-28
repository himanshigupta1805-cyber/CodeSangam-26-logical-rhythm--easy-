"""EDA plots + residual error diagnostics (error clusters, edge cases, heteroskedasticity)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from . import config as C
from .data import load_data, clean_train, drop_leaky_columns, add_base_target
from .features import build_features


def eda(train, out):
    d = build_features(train)
    d = d[d[C.TARGET] > 0]   # fare plots exclude refund rows
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0, 0].hist(d[C.TARGET].clip(upper=d[C.TARGET].quantile(.995)), bins=80, color="#8b1a1a"); ax[0, 0].set_title("total_amount (clipped 99.5%)")
    ax[0, 1].hist(np.log1p(d[C.TARGET]), bins=80, color="#444"); ax[0, 1].set_title("log1p(total_amount)")
    s = d.sample(min(len(d), 8000), random_state=0)
    ax[0, 2].scatter(s[C.DIST].clip(upper=s[C.DIST].quantile(.995)), s[C.TARGET], s=3, alpha=.3); ax[0, 2].set_title("fare vs distance")
    d.groupby("hour")[C.TARGET].mean().plot(ax=ax[1, 0], marker="o"); ax[1, 0].set_title("mean fare by hour")
    d.groupby("dow")[C.TARGET].mean().plot.bar(ax=ax[1, 1]); ax[1, 1].set_title("mean fare by weekday")
    if "RatecodeID" in d:
        d.groupby("RatecodeID")[C.TARGET].mean().plot.bar(ax=ax[1, 2]); ax[1, 2].set_title("mean fare by rate code")
    plt.tight_layout(); plt.savefig(out / "eda_overview.png", dpi=110); plt.close()

    num = d.select_dtypes("number").drop(columns=["PAIR"], errors="ignore")
    corr = num.corr()[C.TARGET].drop(C.TARGET).sort_values(key=abs, ascending=False).head(20)
    plt.figure(figsize=(7, 6)); corr[::-1].plot.barh(color="#8b1a1a"); plt.title("Top |corr| with total_amount")
    plt.tight_layout(); plt.savefig(out / "eda_correlations.png", dpi=110); plt.close()


def residuals(oof, out):
    r = oof.copy(); r["resid"] = r.y_true - r.y_pred; r["abs_err"] = r.resid.abs()
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    ax[0, 0].scatter(r.y_pred, r.y_true, s=3, alpha=.25); m = r.y_true.quantile(.995)
    ax[0, 0].plot([0, m], [0, m], "r--"); ax[0, 0].set_xlim(0, m); ax[0, 0].set_ylim(0, m); ax[0, 0].set_title("Actual vs predicted (OOF)")
    ax[0, 1].scatter(r.y_pred, r.resid, s=3, alpha=.25); ax[0, 1].axhline(0, c="r"); ax[0, 1].set_xlim(0, m); ax[0, 1].set_title("Residual vs predicted (heteroskedasticity)")
    r.groupby("hour").abs_err.mean().plot(ax=ax[1, 0], marker="o"); ax[1, 0].set_title("MAE by hour")
    ax[1, 1].hist(r.resid.clip(-r.resid.abs().quantile(.995), r.resid.abs().quantile(.995)), bins=80); ax[1, 1].set_title("Residual distribution")
    plt.tight_layout(); plt.savefig(out / "residual_diagnostics.png", dpi=110); plt.close()

    # Heteroskedasticity: |resid| vs prediction, plus Breusch-Pagan style test (resid^2 ~ pred)
    rho, p_rho = stats.spearmanr(r.y_pred, r.abs_err)
    lr = stats.linregress(r.y_pred, r.resid ** 2)
    n = len(r); lm = n * lr.rvalue ** 2; p_bp = 1 - stats.chi2.cdf(lm, 1)
    r["pred_bin"] = pd.qcut(r.y_pred, 10, duplicates="drop")
    by_bin = r.groupby("pred_bin", observed=True).agg(rmse=("resid", lambda x: np.sqrt(np.mean(x ** 2))), n=("resid", "size"))

    # Error clusters: worst zone pairs and edge-case slices
    pair = r.groupby(["PU", "DO"]).agg(n=("resid", "size"), mae=("abs_err", "mean"), bias=("resid", "mean"))
    worst_pairs = pair[pair.n >= 5].sort_values("mae", ascending=False).head(15)
    edge = {
        "long_trip(>p99 dist)": r[r[C.DIST] > r[C.DIST].quantile(.99)],
        "very_short(<0.3mi)": r[r[C.DIST] < 0.3],
        "slow(<3mph)": r[(r[C.DIST] / (r.duration_min / 60)) < 3],
        "refund_rows": r[r.is_refund == 1],
        "top1%_fares": r[r.y_true > r.y_true.quantile(.99)],
    }
    lines = ["# Residual diagnostics", "",
             f"- Overall RMSE {np.sqrt(np.mean(r.resid**2)):.4f} | MAE {r.abs_err.mean():.4f} | bias {r.resid.mean():+.4f}",
             f"- Heteroskedasticity: Spearman(|resid|, pred) = {rho:.3f} (p={p_rho:.2g}); BP-style LM p-value = {p_bp:.2g}",
             "  (error grows with fare size: consider log-target / sample weighting)" if rho > .2 else "  (mild)", "",
             "## RMSE by predicted-fare decile", by_bin.to_markdown(), "",
             "## Edge-case slices", pd.DataFrame({k: {"n": len(v), "mae": v.abs_err.mean(), "bias": v.resid.mean()} for k, v in edge.items()}).T.to_markdown(), "",
             "## Worst zone pairs (n>=5)", worst_pairs.to_markdown()]
    (out / "residual_report.md").write_text("\n".join(lines))
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(C.DEFAULT_DATA_DIR)); ap.add_argument("--out-dir", default=str(C.DEFAULT_OUT_DIR))
    a = ap.parse_args(); out = Path(a.out_dir); out.mkdir(exist_ok=True, parents=True)
    train, _, _ = load_data(a.data_dir)
    train, _ = clean_train(drop_leaky_columns(train)); train = add_base_target(train)
    eda(train, out)
    if (out / "oof_predictions.csv").exists():
        residuals(pd.read_csv(out / "oof_predictions.csv"), out)
