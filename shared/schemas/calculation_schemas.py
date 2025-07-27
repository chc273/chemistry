"""Shared schemas for calculation requests and results."""

from typing import Any

from pydantic import BaseModel, Field


class CalculationRequest(BaseModel):
    """Schema for quantum chemistry calculation requests."""

    method: str = Field(description="Calculation method (hf, dft, etc.)")
    basis_set: str = Field(description="Basis set name")
    system_id: str = Field(description="Unique system identifier")
    properties: list[str] = Field(default=[], description="Properties to calculate")
    options: dict[str, Any] = Field(default={}, description="Additional options")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class CalculationResult(BaseModel):
    """Schema for calculation results."""

    request_id: str = Field(description="Original request identifier")
    energy: float = Field(description="Total energy in Hartree")
    converged: bool = Field(description="Whether calculation converged")
    properties: dict[str, Any] = Field(default={}, description="Calculated properties")
    metadata: dict[str, Any] = Field(default={}, description="Calculation metadata")

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
