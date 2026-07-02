import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

DATA_DIR = "/Users/mhemmett/Seismology/axial-splitting-ml/data"
df = pd.read_csv(f"{DATA_DIR}/axec1_pgv_2015_2021.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
df = df.sort_values("timestamp").reset_index(drop=True)

t = mdates.date2num(df["timestamp"].dt.to_pydatetime())
pgv = df["pgv_counts"].values

fig, ax = plt.subplots(figsize=(14, 5))

ax.scatter(t, pgv, s=1, alpha=0.3, color="steelblue", rasterized=True, label="PGV per event")

# 30-day rolling median (by index window ~4500 events)
pgv_series = pd.Series(pgv)
rolling = pgv_series.rolling(4500, min_periods=100, center=True).median()
ax.plot(t, rolling.values, color="crimson", linewidth=1.5, label="Rolling median (~30d)")

# Eruption markers
for date_str, label in [("2015-04-24", "Apr 2015 eruption")]:
    t_mark = mdates.date2num(pd.Timestamp(date_str).to_pydatetime())
    ax.axvline(t_mark, color="orange", linestyle="--", linewidth=1.2, alpha=0.9)
    ax.text(t_mark, ax.get_ylim()[1], label, rotation=90, va="top", ha="right", fontsize=8, color="darkorange")

ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlabel("Date")
ax.set_ylabel("PGV (counts)")
ax.set_title("AXEC1 Peak Ground Velocity 2015–2021")
ax.legend(markerscale=5)
plt.tight_layout()
plt.savefig(f"{DATA_DIR}/axec1_pgv_2015_2021.png", dpi=150)
print("Saved plot.")
plt.show()
