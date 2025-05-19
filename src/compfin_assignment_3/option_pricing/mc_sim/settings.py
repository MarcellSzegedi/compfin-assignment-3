"""Class for collecting and validate model settings."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class BSModelSettings(BaseModel):
    """Settings for configuring the Black-Scholes model simulation.

    Attributes:
        s_0 (float): The starting price of the underlying asset.
        sigma (float): The volatility of the underlying asset.
        t_end (float): The upper bound of the time interval for the simulated stochastic process.
        drift (float): The drift of the underlying asset.
        risk_free_rate (float): The risk-free interest rate.
        n_trajectories (int): The number of trajectories to simulate.
        alpha (float): Significance level.
        strike (float): The strike price of the option.
        step_size (Optional[float]): The step size for the Euler scheme. Must be > 0.
        num_steps (Optional[int]): The number of steps for the Euler scheme. Must be > 0.

    """

    s_0: float = Field(..., gt=0)
    sigma: float = Field(..., gt=0)
    t_end: float = Field(..., ge=0)
    drift: float = Field(..., ge=0)
    strike: float = Field(..., gt=0)
    risk_free_rate: float = Field(..., ge=0)
    n_trajectories: int = Field(..., ge=1)
    alpha: float = Field(..., gt=0, lt=1)
    step_size: Optional[float] = Field(default=None, gt=0)
    num_steps: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_step_size_and_num_steps(self) -> "BSModelSettings":
        """Validate the step_size and num_steps fields."""
        if self.step_size is None and self.num_steps is None:
            raise ValueError("Either step_size or num_steps should be specified.")
        if self.step_size is not None and self.num_steps is not None:
            raise ValueError("Only one of step_size or num_steps should be specified.")
        return self

    @model_validator(mode="after")
    def set_step_values(self) -> "BSModelSettings":
        """Set the step_size and num_steps fields based on the specified scheme."""
        if self.step_size is None and self.num_steps is not None:
            self.step_size = self.t_end / self.num_steps
        if self.step_size is not None and self.num_steps is None:
            self.num_steps = int(self.t_end / self.step_size)
        return self

    @model_validator(mode="after")
    def comp_risk_free_rate_to_drift(self) -> "BSModelSettings":
        """Check that the risk-free rate is consistent with the drift."""
        if self.risk_free_rate != self.drift:
            print("Warning: risk-free rate is different from drift.")
        return self
