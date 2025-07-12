import typer

app = typer.Typer()


@app.command()
def main(n1: int, n2: int) -> None:
    """Add the arguments and print the result."""
