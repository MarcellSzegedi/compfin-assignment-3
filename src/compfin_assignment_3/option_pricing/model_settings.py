"""Class for collecting and validate model settings."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class HestonModelSettings(BaseModel):
    """Settings for configuring the Heston model simulation.

    Attributes:
        s_0 (float): The starting price of the underlying asset.
        v_0 (float): The starting variance of the underlying asset.
        t_end (float): The upper bound of the time interval for the simulated stochastic process.
        drift (float): The drift of the underlying asset.
        kappa (float): The rate at which volatility reverts to its long-term mean (theta).
        theta (float): The long-term mean of the volatility.
        vol_of_vol (float): The volatility of volatility.
        stoc_inc_corr (float): The correlation between the stochastic increments of the underlying
                                asset and the volatility process.
        risk_free_rate (float): The risk-free interest rate.
        n_trajectories (int): The number of trajectories to simulate.
        alpha (float): Significance level.
        strike (float): The strike price of the option.
        step_size (Optional[float]): The step size for the Euler scheme. Must be > 0.
        num_steps (Optional[int]): The number of steps for the Euler scheme. Must be > 0.

    """

    s_0: float = Field(..., gt=0)
    v_0: float = Field(..., ge=0)
    t_end: float = Field(..., ge=0)
    drift: float = Field(..., ge=0)
    kappa: float = Field(..., ge=0)
    theta: float = Field(..., ge=0)
    vol_of_vol: float = Field(..., ge=0)
    stoc_inc_corr: float = Field(..., ge=-1, le=1)
    strike: float = Field(..., gt=0)
    risk_free_rate: float = Field(..., ge=0)
    n_trajectories: int = Field(..., ge=1)
    alpha: float = Field(..., gt=0, lt=1)
    step_size: Optional[float] = Field(default=None, gt=0)
    num_steps: Optional[int] = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_step_size_and_num_steps(self) -> "HestonModelSettings":
        """Validate the step_size and num_steps fields."""
        if self.step_size is None and self.num_steps is None:
            raise ValueError("Either step_size or num_steps should be specified.")
        if self.step_size is not None and self.num_steps is not None:
            raise ValueError("Only one of step_size or num_steps should be specified.")
        return self

    @model_validator(mode="after")
    def set_step_values(self) -> "HestonModelSettings":
        """Set the step_size and num_steps fields based on the specified scheme."""
        if self.step_size is None and self.num_steps is not None:
            self.step_size = self.t_end / self.num_steps
        if self.step_size is not None and self.num_steps is None:
            self.num_steps = int(self.t_end / self.step_size)
        return self
