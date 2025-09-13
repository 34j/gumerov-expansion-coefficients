import csv
from pathlib import Path

import pandas as pd
import seaborn as sns
import torch as torch_original
import typer
from array_api_compat import torch
from cm_time import timer
from numba import threading_layer
from rich import print

from gumerov_expansion_coefficients import translational_coefficients

app = typer.Typer()

# estimated memory: 4 byte  * size * (2 * n_end) ** 4


def _max_n_end(memory_bytes: int, size: int, is_single: bool) -> int:
    return int((memory_bytes / ((2 if is_single else 4) * size)) ** 0.25 / 2)


@app.command()
def benchmark(
    devices: str = "cuda,cpu",
    dtypes: str = "float32,float64",
    n_loops: int = 3,
    max_memory_gb: int = 1,
) -> None:
    with Path("timing_results.csv").open("w") as f:
        print(f"Threading layer chosen: {threading_layer()}")
        writer = csv.DictWriter(
            f, fieldnames=["backend", "device", "dtype", "size", "n_end", "time"]
        )
        writer.writeheader()
        for name, xp in [
            ("torch", torch),
        ]:
            for device in devices.split(","):
                if name == "numpy" and device == "cuda":
                    continue
                for dtype in [getattr(xp, dtype_str) for dtype_str in dtypes.split(",")]:
                    try:
                        for size in 4 ** xp.arange(0, 6):
                            for n_end in range(
                                2,
                                min(
                                    25,
                                    _max_n_end(
                                        max_memory_gb * 1024 * 1024 * 1024, size, dtype == "float32"
                                    ),
                                ),
                                2,
                            ):
                                kr = xp.arange(size, dtype=dtype, device=device)
                                theta = xp.arange(size, dtype=dtype, device=device)
                                phi = xp.arange(size, dtype=dtype, device=device)
                                for i in range(1 + n_loops):
                                    with timer() as t:
                                        coef = translational_coefficients(
                                            kr=kr,
                                            theta=theta,
                                            phi=phi,
                                            same=False,
                                            n_end=n_end,
                                        )
                                        str(coef[0, 0])
                                        del coef
                                    if i == 0:
                                        continue
                                    result = {
                                        "backend": name,
                                        "device": device,
                                        "dtype": str(dtype).split(".")[-1].split("'")[0],
                                        "size": int(size),
                                        "n_end": n_end,
                                        "time": t.elapsed,
                                    }
                                    writer.writerow(result)
                                    print(result)
                                torch_original.cuda.empty_cache()
                    except Exception as e:
                        raise e


@app.command()
def plot() -> None:
    df = pd.read_csv("timing_results.csv")
    hue_name = "Backend, Device"
    df[hue_name] = df[["backend", "device"]].agg(", ".join, axis=1)
    hue_unique = df[hue_name].unique()
    sns.set_theme()
    g = sns.relplot(
        data=df,
        x="n_end",
        y="time",
        hue=hue_name,
        style=hue_name,
        col="size",
        row="dtype",
        kind="line",
        markers={k: "o" if "cuda" in k else "X" for k in hue_unique},
        height=3,
        aspect=1,
    )
    g.set_xlabels("N - 1")
    g.set_ylabels("Time (s)")
    g.set(yscale="log")
    g.savefig("timing_results.jpg", dpi=300, bbox_inches="tight")
    g.savefig("timing_results.png", dpi=300, bbox_inches="tight")
    g.savefig("timing_results.eps", bbox_inches="tight")
