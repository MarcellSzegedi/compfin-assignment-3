"""Main entry of the CLI application."""

import typer

app = typer.Typer()
__version__ = "0.1.0"


@app.command()
def version():
    """Prints the version of the CLI application."""
    typer.echo(f"CompFin Assignment 3 CLI v{__version__}")
