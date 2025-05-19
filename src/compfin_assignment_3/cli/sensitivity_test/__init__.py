"""Sensitivity plot collection."""

import typer

from .risk_free_rate import app as risk_free_rate_app
from .time_mature import app as time_mature_app
from .volatility import app as volatility_app

app = typer.Typer()


app.add_typer(volatility_app)
app.add_typer(risk_free_rate_app)
app.add_typer(time_mature_app)
