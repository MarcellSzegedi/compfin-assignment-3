"""Class for collecting and validate model settings."""

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class HeatEquationSettings(BaseModel):
    """Class for collecting and validate model settings."""

    s_0: float = Field(..., gt=0)
    sigma: float = Field(..., gt=0)
    t_end: float = Field(..., ge=0)
    strike: float = Field(..., gt=0)
    risk_free_rate: float = Field(..., ge=0)
    n_step_t: Optional[int] = Field(default=None, ge=1)
    n_step_x: Optional[int] = Field(default=None, ge=1)
    step_size_t: Optional[float] = Field(default=None, gt=0)
    step_size_x: Optional[float] = Field(default=None, gt=0)
    min_s: float = Field(..., gt=0)
    max_s: float = Field(..., gt=0)
    alpha: Optional[float] = Field(default=None, init=False)
    beta: Optional[float] = Field(default=None, init=False)

    @model_validator(mode="after")
    def validate_step_size_and_num_steps(self) -> "HeatEquationSettings":
        """Validate the step_size and num_steps fields."""
        if self.step_size_t is None and self.n_step_t is None:
            raise ValueError("Either step_size or num_steps should be specified for time.")
        if self.step_size_t is not None and self.n_step_t is not None:
            raise ValueError("Only one of step_size or num_steps should be specified for time")

        if self.step_size_x is None and self.n_step_x is None:
            raise ValueError(
                "Either step_size or num_steps should be specified for the asset price."
            )
        if self.step_size_x is not None and self.n_step_x is not None:
            raise ValueError(
                "Only one of step_size or num_steps should be specified for the asset price."
            )
        return self

    @model_validator(mode="after")
    def check_min_max_s(self) -> "HeatEquationSettings":
        """Check that the min_s and max_s fields are consistent."""
        if self.min_s >= self.max_s:
            raise ValueError("min_s should be smaller than max_s.")
        return self

    @model_validator(mode="after")
    def set_step_values(self) -> "HeatEquationSettings":
        """Set the step_size and num_steps fields based on the specified scheme."""
        if self.step_size_t is None and self.n_step_t is not None:
            self.step_size_t = self.t_end / self.n_step_t
        if self.step_size_t is not None and self.n_step_t is None:
            self.n_step_t = int(self.t_end / self.step_size_t)
        if self.step_size_x is None and self.n_step_x is not None:
            self.step_size_x = (self.max_s - self.min_s) / self.n_step_x
        if self.step_size_x is not None and self.n_step_x is None:
            self.n_step_x = int(self.t_end / self.step_size_x)
        return self

    @model_validator(mode="after")
    def set_variable_transformation_params(self) -> "HeatEquationSettings":
        """Set the additional parameters for the variable transformation (alpha and beta)."""
        self.alpha = 0.5 - self.risk_free_rate / self.sigma**2
        self.beta = self.alpha * self.risk_free_rate + self.sigma**2 / 2 * (
            self.alpha**2 - self.alpha
        )
        return self
