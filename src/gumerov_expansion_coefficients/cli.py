import pandas as pd
import typer
from array_api_compat import numpy, torch
from cm_time import timer
from rich import print

from gumerov_expansion_coefficients import translational_coefficients

app = typer.Typer()


@app.command()
def main() -> None:
    results = []
    for name, xp in [
        ("numpy", numpy),
        ("torch", torch),
        # ("jax", jnp),
    ]:
        for device in ["cpu", "cuda"]:
            try:
                for size in 2 ** xp.arange(8, 12):
                    for n_end in range(1, 15):
                        kr = xp.arange(size, dtype=xp.float32, device=device)
                        theta = xp.arange(size, dtype=xp.float32, device=device)
                        phi = xp.arange(size, dtype=xp.float32, device=device)
                        with timer() as t:
                            translational_coefficients(
                                kr=kr,
                                theta=theta,
                                phi=phi,
                                same=True,
                                n_end=n_end,
                            )
                        results.append(
                            {
                                "backend": name,
                                "size": size,
                                "n_end": n_end,
                                "time": t.elapsed,
                            }
                        )
                        print(results[-1])
            except Exception as e:
                print(e)
    df = pd.DataFrame(results)
    df.to_csv("timing_results.csv", index=False)
