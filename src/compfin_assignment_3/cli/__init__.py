"""Main entry of the CLI application."""

import typer

from .option_pricing.binary_option_heat_eq import app as binary_option_heat_eq_app
from .option_pricing.binary_option_mc_sim import app as binary_option_mc_sim_app

app = typer.Typer()
__version__ = "0.1.0"

app.add_typer(binary_option_mc_sim_app)
app.add_typer(binary_option_heat_eq_app)


@app.command()
def version():
    """Prints the version of the CLI application."""
    typer.echo(f"CompFin Assignment 3 CLI v{__version__}")
