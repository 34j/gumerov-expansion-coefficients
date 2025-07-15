import csv
from pathlib import Path

import pandas as pd
import seaborn as sns
import typer
from array_api_compat import numpy, torch
from cm_time import timer
from rich import print

from gumerov_expansion_coefficients import translational_coefficients

app = typer.Typer()


@app.command()
def benchmark() -> None:
    with Path("timing_results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "device", "size", "n_end", "time"])
        writer.writeheader()
        for name, xp in [
            ("numpy", numpy),
            ("torch", torch),
            # ("jax", jnp),
        ]:
            for device in ["cuda", "cpu"]:
                for dtype in [xp.float32, xp.float64]:
                    try:
                        for size in 2 ** xp.arange(10, 14):
                            for n_end in range(8, 14):
                                kr = xp.arange(size, dtype=dtype, device=device)
                                theta = xp.arange(size, dtype=dtype, device=device)
                                phi = xp.arange(size, dtype=dtype, device=device)
                                with timer() as t:
                                    translational_coefficients(
                                        kr=kr,
                                        theta=theta,
                                        phi=phi,
                                        same=True,
                                        n_end=n_end,
                                    )
                                result = {
                                    "backend": name,
                                    "device": device,
                                    "size": int(size),
                                    "n_end": n_end,
                                    "time": t.elapsed,
                                }
                                writer.writerow(result)
                                print(result)
                    except Exception as e:
                        print(e)


@app.command()
def plot() -> None:
    df = pd.read_csv("timing_results.csv")
    df["hue"] = df[["backend", "device"]].agg("-".join, axis=1)
    hue_unique = df["hue"].unique()
    sns.set_theme()
    g = sns.relplot(
        data=df,
        x="n_end",
        y="time",
        hue="hue",
        style="hue",
        col="size",
        row="dtype",
        kind="line",
        markers={k: "o" if "cuda" in k else "X" for k in hue_unique},
    )
    g.set_ylabels("Time (s)")
    g.set(xscale="log", yscale="log")
    g.savefig("timing_results.png", dpi=300, bbox_inches="tight")
