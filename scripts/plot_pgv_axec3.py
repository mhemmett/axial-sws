import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DATA_DIR = "/Users/mhemmett/Seismology/axial-splitting-ml/data"
df = pd.read_csv(f"{DATA_DIR}/axec3_pgv_2015_2021.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
df = df.sort_values("timestamp").reset_index(drop=True)

t = mdates.date2num(np.array(df["timestamp"].dt.to_pydatetime()))
pgv = df["pgv_ms"].values * 1e9  # convert m/s to nm/s

fig, ax = plt.subplots(figsize=(14, 5))

ax.scatter(t, pgv, s=0.5, alpha=0.2, color="steelblue", rasterized=True, label="PGV per event")

# Rolling median over ~1000 events (~30 days at typical rate)
rolling = pd.Series(pgv).rolling(1000, min_periods=50, center=True).median()
ax.plot(t, rolling.values, color="crimson", linewidth=1.5, label="Rolling median (~1000 events)")

# Eruption markers
for date_str, label in [("2015-04-24", "Apr 2015 eruption"), ("2021-12-02", "Dec 2021 eruption")]:
    t_mark = mdates.date2num(pd.Timestamp(date_str).to_pydatetime())
    ax.axvline(t_mark, color="orange", linestyle="--", linewidth=1.2, alpha=0.9)
    ymax = ax.get_ylim()[1] or 1
    ax.text(t_mark, ymax * 0.95, label, rotation=90, va="top", ha="right", fontsize=8, color="darkorange",
            transform=ax.get_xaxis_transform())

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Date")
ax.set_ylabel("PGV (nm/s)")
ax.set_title("AXEC3 Peak Ground Velocity 2015–2021 (n=131,854 events)")
ax.legend(markerscale=8)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/axec3_pgv_2015_2021.png", dpi=150)
print("Saved plot.")

# Per-year summary
df["pgv_nms"] = df["pgv_ms"] * 1e9
df["year"] = df["timestamp"].dt.year
print(df.groupby("year")["pgv_nms"].agg(["mean", "median", "count"]).round(1))

plt.show()
