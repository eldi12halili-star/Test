"""Albanian housing price forecasting model (annual, to 2035).

The model predicts the year-on-year log growth of the average residential price
(EUR per m2) from lagged price momentum and macroeconomic drivers, then rolls
the prediction forward recursively under macro scenarios. Prediction intervals
come from bootstrapping walk-forward residuals along each simulated path.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "growth_lag1",
    "gdp_growth_pct",
    "real_rate_pct",
    "tourism_growth_pct",
    "eur_all_change_pct",
    "remittance_growth_pct",
    "valuation_gap",
]
FIRST_TEST_YEAR = 2018


def build_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw yearly table into model features and the growth target."""
    df = hist.sort_values("year").reset_index(drop=True).copy()
    df["growth"] = np.log(df["price_eur_m2"]).diff()
    df["growth_lag1"] = df["growth"].shift(1)
    df["real_rate_pct"] = df["policy_rate_pct"] - df["inflation_pct"]
    df["tourism_growth_pct"] = 100 * np.log(df["tourist_arrivals_mn"]).diff()
    df["eur_all_change_pct"] = 100 * np.log(df["eur_all"]).diff()
    df["remittance_growth_pct"] = 100 * np.log(df["remittances_bn_eur"]).diff()
    # Error-correction anchor: log(price / GDP per capita in EUR) at the end of
    # last year, relative to its long-run mean. Prices well above incomes tend
    # to grow more slowly afterwards, which stops forecasts drifting forever.
    df["log_price_income"] = np.log(df["price_eur_m2"] / df["gdp_per_capita_eur"])
    df["valuation_gap"] = (df["log_price_income"] - df["log_price_income"].mean()).shift(1)
    return df


def make_models() -> dict:
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=3.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=400, max_depth=3, min_samples_leaf=2, random_state=0
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0
        ),
    }


def predict_ensemble(models: dict, X: pd.DataFrame) -> np.ndarray:
    return np.mean([m.predict(X) for m in models.values()], axis=0)


def walk_forward(df: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window backtest: train on years < t, predict year t."""
    data = df.dropna(subset=FEATURES + ["growth"])
    rows = []
    for year in data.loc[data.year >= FIRST_TEST_YEAR, "year"]:
        train, test = data[data.year < year], data[data.year == year]
        models = make_models()
        preds = {}
        for name, m in models.items():
            m.fit(train[FEATURES], train["growth"])
            preds[name] = float(m.predict(test[FEATURES])[0])
        preds["ensemble"] = float(np.mean(list(preds.values())))
        preds["naive_last_growth"] = float(test["growth_lag1"].iloc[0])
        rows.append({"year": int(year), "actual": float(test["growth"].iloc[0]), **preds})
    return pd.DataFrame(rows)


def backtest_metrics(bt: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    prev_price = df.set_index("year")["price_eur_m2"].shift(1)
    out = []
    for col in ["ridge", "random_forest", "gradient_boosting", "ensemble", "naive_last_growth"]:
        err = bt[col] - bt["actual"]
        p_prev = prev_price.loc[bt.year].to_numpy()
        price_err = p_prev * np.exp(bt[col]) - p_prev * np.exp(bt["actual"])
        out.append({
            "model": col,
            "growth_MAE_pp": 100 * err.abs().mean(),
            "price_MAE_eur": np.abs(price_err).mean(),
            "price_MAPE_pct": 100 * np.mean(np.abs(price_err) / (p_prev * np.exp(bt["actual"]))),
        })
    return pd.DataFrame(out).sort_values("price_MAE_eur")


def forecast(df: pd.DataFrame, scen: pd.DataFrame, residuals: np.ndarray,
             n_sims: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Recursive scenario forecast with residual-bootstrap intervals."""
    data = df.dropna(subset=FEATURES + ["growth"])
    models = make_models()
    for m in models.values():
        m.fit(data[FEATURES], data["growth"])

    rng = np.random.default_rng(seed)
    residuals = residuals - residuals.mean()  # centre: intervals around the point path
    last = df.iloc[-1]
    hist_logp = list(np.log(df["price_eur_m2"]))
    pi_mean = df["log_price_income"].mean()
    results = []
    for name, s in scen.groupby("scenario"):
        s = s.sort_values("year")
        years = s["year"].to_numpy()
        # paths[k] holds log prices for simulation k; path 0 is the point forecast.
        logp = np.tile(hist_logp, (n_sims + 1, 1))
        g_prev = np.full(n_sims + 1, last["growth"])
        log_inc = np.log(last["gdp_per_capita_eur"])
        for _, r in s.iterrows():
            gap = (logp[:, -1] - log_inc) - pi_mean
            # Income per capita in EUR: real growth - population growth + inflation,
            # adjusted for the lek/euro rate (fewer lek per euro = higher EUR income).
            log_inc += (r.gdp_growth_pct - r.population_growth_pct
                        + r.inflation_pct - r.eur_all_change_pct) / 100
            X = pd.DataFrame({
                "growth_lag1": g_prev,
                "gdp_growth_pct": r.gdp_growth_pct,
                "real_rate_pct": r.policy_rate_pct - r.inflation_pct,
                "tourism_growth_pct": r.tourism_growth_pct,
                "eur_all_change_pct": r.eur_all_change_pct,
                "remittance_growth_pct": r.remittance_growth_pct,
                "valuation_gap": gap,
            })[FEATURES]
            g = predict_ensemble(models, X)
            noise = rng.choice(residuals, size=n_sims + 1)
            noise[0] = 0.0
            g = g + noise
            logp = np.column_stack([logp, logp[:, -1] + g])
            g_prev = g
        prices = np.exp(logp[:, len(hist_logp):])
        for j, y in enumerate(years):
            sims = prices[1:, j]
            results.append({
                "scenario": name,
                "year": int(y),
                "forecast_eur_m2": round(float(prices[0, j])),
                "p10": round(float(np.percentile(sims, 10))),
                "p50": round(float(np.percentile(sims, 50))),
                "p90": round(float(np.percentile(sims, 90))),
            })
    out = pd.DataFrame(results)
    out["growth_vs_prev_pct"] = (
        out.groupby("scenario")["forecast_eur_m2"].pct_change().mul(100).round(1)
    )
    first = out.year == out.year.min()
    out.loc[first, "growth_vs_prev_pct"] = round(
        100 * (out.loc[first, "forecast_eur_m2"] / last["price_eur_m2"] - 1), 1)
    return out


def feature_importance(df: pd.DataFrame) -> pd.Series:
    data = df.dropna(subset=FEATURES + ["growth"])
    rf = make_models()["random_forest"].fit(data[FEATURES], data["growth"])
    return pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)


def plot(df: pd.DataFrame, fc: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": "#2a6fdb", "optimistic": "#1a9e5c", "pessimistic": "#d1495b"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(df.year, df.price_eur_m2, color="#222", lw=2.2, label="Historical")
    for name, g in fc.groupby("scenario"):
        yrs = [df.year.iloc[-1]] + list(g.year)
        ax.plot(yrs, [df.price_eur_m2.iloc[-1]] + list(g.forecast_eur_m2),
                color=colors.get(name), lw=2, ls="--", label=f"{name.title()} forecast")
        if name == "baseline":
            ax.fill_between(g.year, g.p10, g.p90, color=colors[name], alpha=0.15,
                            label="Baseline 80% interval")
    ax.set_title("Albania (Tirana) average residential price, EUR/m², forecast to 2035")
    ax.set_ylabel("EUR per m²")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=ROOT / "data" / "albania_housing.csv")
    ap.add_argument("--scenarios", default=ROOT / "data" / "scenarios.csv")
    ap.add_argument("--out", default=ROOT / "outputs")
    ap.add_argument("--sims", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    df = build_features(pd.read_csv(args.data))
    scen = pd.read_csv(args.scenarios)

    bt = walk_forward(df)
    metrics = backtest_metrics(bt, df)
    residuals = (bt["actual"] - bt["ensemble"]).to_numpy()
    fc = forecast(df, scen, residuals, n_sims=args.sims)
    imp = feature_importance(df)

    bt.to_csv(out / "backtest.csv", index=False)
    metrics.to_csv(out / "backtest_metrics.csv", index=False)
    fc.to_csv(out / "forecast_2026_2035.csv", index=False)
    plot(df, fc, out / "forecast_2035.png")

    pd.set_option("display.width", 120)
    print("Walk-forward backtest (%d-%d):" % (bt.year.min(), bt.year.max()))
    print(metrics.round(2).to_string(index=False))
    print("\nFeature importance (random forest):")
    print(imp.round(3).to_string())
    print("\nForecast (EUR/m2):")
    print(fc.pivot(index="year", columns="scenario", values="forecast_eur_m2").to_string())
    b = fc[fc.scenario == "baseline"].set_index("year")
    print("\nBaseline 80%% interval 2035: %d - %d EUR/m2" % (b.loc[2035, "p10"], b.loc[2035, "p90"]))


if __name__ == "__main__":
    main()
